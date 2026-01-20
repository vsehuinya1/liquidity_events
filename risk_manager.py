# risk_manager.py
"""
Phase 2 Risk Engine: The "No" Layer.
Tuned based on Backtest 2026-01-17:
- REMOVED: Session Scaling (Proven to reduce Expectancy).
- KEPT: Correlation Gating (Systemic Risk protection).
- MODIFIED: Daily Limit loose (-10R) for Catastrophic Protection only.

Architecture:
    - RiskState: Explicit, serializable state container
    - Pure validation functions for each risk check
    - All state transitions explicit and logged
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional, Any, Set, FrozenSet


# =============================================================================
# CONSTANTS
# =============================================================================

# Default correlation bucket mappings (frozen)
DEFAULT_ASSET_BUCKETS: Dict[str, str] = {
    'SOLUSDT': 'L1',
    'ETHUSDT': 'L1',
    'AVAXUSDT': 'L1',
    'SUIUSDT': 'L1',
    'DOGEUSDT': 'MEME',
    'PEPEUSDT': 'MEME'
}


# =============================================================================
# STATE CONTAINER
# =============================================================================

@dataclass
class RiskState:
    """
    Explicit, serializable risk state.
    All fields are deterministic and restart-safe.
    """
    daily_pnl_r: float = 0.0
    current_date: date = field(default_factory=lambda: datetime.utcnow().date())
    active_buckets: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# PURE FUNCTIONS: DATE HANDLING
# =============================================================================

def is_new_day(current_date: date) -> bool:
    """Check if current date has changed (new trading day)."""
    today = datetime.utcnow().date()
    return today > current_date


def get_today() -> date:
    """Get current UTC date."""
    return datetime.utcnow().date()


# =============================================================================
# PURE FUNCTIONS: RISK CHECKS
# =============================================================================

def check_circuit_breaker(daily_pnl_r: float, limit_r: float) -> bool:
    """
    Check if daily loss limit has been breached.
    Returns True if BLOCKED (limit hit), False if OK.
    """
    return daily_pnl_r <= limit_r


def check_correlation_limit(
    symbol: str,
    asset_buckets: Dict[str, str],
    active_buckets: Dict[str, int],
    max_correlated: int
) -> tuple:
    """
    Check if correlation bucket is full.
    Returns (is_blocked, bucket_name, current_count).
    """
    bucket = asset_buckets.get(symbol, 'OTHER')
    current_active = active_buckets.get(bucket, 0)
    is_blocked = current_active >= max_correlated
    return is_blocked, bucket, current_active


def get_bucket_for_symbol(symbol: str, asset_buckets: Dict[str, str]) -> str:
    """Get correlation bucket for a symbol."""
    return asset_buckets.get(symbol, 'OTHER')


# =============================================================================
# PURE FUNCTIONS: STATE TRANSITIONS
# =============================================================================

def reset_daily_state(state: RiskState) -> RiskState:
    """
    Create new state with daily PnL reset.
    Returns new state object (does not mutate input).
    """
    return RiskState(
        daily_pnl_r=0.0,
        current_date=get_today(),
        active_buckets=state.active_buckets.copy()
    )


def increment_bucket_count(active_buckets: Dict[str, int], bucket: str) -> Dict[str, int]:
    """
    Increment bucket count.
    Returns new dict (does not mutate input).
    """
    result = active_buckets.copy()
    if bucket not in result:
        result[bucket] = 0
    result[bucket] += 1
    return result


def decrement_bucket_count(active_buckets: Dict[str, int], bucket: str) -> Dict[str, int]:
    """
    Decrement bucket count (minimum 0).
    Returns new dict (does not mutate input).
    """
    result = active_buckets.copy()
    if bucket in result and result[bucket] > 0:
        result[bucket] -= 1
    return result


def add_pnl_to_daily(current_pnl_r: float, delta_r: float) -> float:
    """Add PnL delta to daily total."""
    return current_pnl_r + delta_r


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_risk_logging() -> logging.Logger:
    """Configure risk manager logger."""
    logger = logging.getLogger('RiskManager')
    logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# MAIN RISK MANAGER CLASS
# =============================================================================

class RiskManager:
    """
    Risk gatekeeper for trade signals.
    
    Responsibilities:
        - Circuit breaker: Block trading after daily loss limit
        - Correlation guard: Limit exposure to correlated assets
        - State tracking: Daily PnL, active positions per bucket
    
    All risk checks are fail-closed (reject on uncertainty).
    """
    
    def __init__(
        self,
        daily_loss_limit_r: float = -10.0,
        max_active_correlated: int = 1
    ):
        self.logger = setup_risk_logging()
        
        # Configuration (immutable after init)
        self.DAILY_LOSS_LIMIT_R = daily_loss_limit_r
        self.MAX_ACTIVE_CORRELATED = max_active_correlated
        
        # Asset bucket mappings
        self.asset_buckets = DEFAULT_ASSET_BUCKETS.copy()
        
        # Initialize state
        initial_buckets = {bucket: 0 for bucket in set(self.asset_buckets.values())}
        self.state = RiskState(
            daily_pnl_r=0.0,
            current_date=get_today(),
            active_buckets=initial_buckets
        )
        
        self.logger.info("🛡️ Risk Manager Initialized (Tuned v1.1)")
        self.logger.info(f"Circuit Breaker: {self.DAILY_LOSS_LIMIT_R}R | Max Correlated: {self.MAX_ACTIVE_CORRELATED}")

    # =========================================================================
    # PROPERTIES (for backward compatibility)
    # =========================================================================
    
    @property
    def daily_pnl_r(self) -> float:
        """Backward-compatible accessor."""
        return self.state.daily_pnl_r
    
    @daily_pnl_r.setter
    def daily_pnl_r(self, value: float) -> None:
        """Backward-compatible setter."""
        self.state.daily_pnl_r = value
    
    @property
    def current_date(self) -> date:
        """Backward-compatible accessor."""
        return self.state.current_date
    
    @current_date.setter
    def current_date(self, value: date) -> None:
        """Backward-compatible setter."""
        self.state.current_date = value
    
    @property
    def active_buckets(self) -> Dict[str, int]:
        """Backward-compatible accessor."""
        return self.state.active_buckets
    
    @active_buckets.setter
    def active_buckets(self, value: Dict[str, int]) -> None:
        """Backward-compatible setter."""
        self.state.active_buckets = value

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _maybe_reset_daily(self) -> None:
        """
        Reset daily state if new day detected.
        Side effect: Mutates self.state.
        """
        if is_new_day(self.state.current_date):
            old_date = self.state.current_date
            self.state = reset_daily_state(self.state)
            self.logger.info(f"New Day Detected ({self.state.current_date}). Daily PnL Reset.")

    # =========================================================================
    # PNL TRACKING
    # =========================================================================
    
    def update_pnl(self, pnl_r: float) -> None:
        """
        Update Daily PnL state from Executor feedback.
        Side effect: Mutates self.state.daily_pnl_r.
        """
        self._maybe_reset_daily()
        self.state.daily_pnl_r = add_pnl_to_daily(self.state.daily_pnl_r, pnl_r)
        self.logger.info(f"Daily PnL Update: {self.state.daily_pnl_r:.2f}R")

    # =========================================================================
    # CORRELATION TRACKING
    # =========================================================================
    
    def register_entry(self, symbol: str) -> None:
        """
        Register confirmed entry to update correlation state.
        Side effect: Mutates self.state.active_buckets.
        """
        bucket = get_bucket_for_symbol(symbol, self.asset_buckets)
        self.state.active_buckets = increment_bucket_count(self.state.active_buckets, bucket)

    def register_exit(self, symbol: str) -> None:
        """
        Register confirmed exit to update correlation state.
        Side effect: Mutates self.state.active_buckets.
        """
        bucket = get_bucket_for_symbol(symbol, self.asset_buckets)
        self.state.active_buckets = decrement_bucket_count(self.state.active_buckets, bucket)

    # =========================================================================
    # SIGNAL VALIDATION
    # =========================================================================
    
    def validate_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The Main Gatekeeper.
        
        Pipeline:
            1. Reset daily state if new day
            2. Circuit breaker check (daily loss limit)
            3. Correlation guard check (bucket limit)
        
        Returns: Signal (passed through) OR None (rejected).
        """
        self._maybe_reset_daily()
        symbol = signal['symbol']
        
        # CHECK 1: Circuit breaker (daily loss limit)
        if check_circuit_breaker(self.state.daily_pnl_r, self.DAILY_LOSS_LIMIT_R):
            self.logger.warning(
                f"🛑 REJECTED {symbol}: Daily Circuit Breaker Hit ({self.state.daily_pnl_r}R)"
            )
            return None  # Early return: daily limit breached
        
        # CHECK 2: Correlation guard
        is_blocked, bucket, current_count = check_correlation_limit(
            symbol=symbol,
            asset_buckets=self.asset_buckets,
            active_buckets=self.state.active_buckets,
            max_correlated=self.MAX_ACTIVE_CORRELATED
        )
        
        if is_blocked:
            self.logger.warning(
                f"🛑 REJECTED {symbol}: Correlation Bucket '{bucket}' Full "
                f"({current_count}/{self.MAX_ACTIVE_CORRELATED})"
            )
            return None  # Early return: correlation limit reached
        
        # CHECK 3: Session scaling (REMOVED based on 2026-01-17 Backtest)
        # Original logic removed. Asia session proved profitable.
        
        # PASSED: Return original signal unchanged
        return signal
