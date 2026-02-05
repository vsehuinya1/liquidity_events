# live_event_detector_gem.py

import pandas as pd
import numpy as np
from collections import deque
import logging
from datetime import datetime
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import os
from telegram_bot import TelegramBot

# ============================================================================
# LIVE EVENT DETECTOR GEM (v1.3.0) - OPTION F (FUSION) - REFACTORED
# ============================================================================
# Features:
# - Dynamic Filtering (Strict Base / Loose Burst)
# - Burst Mode State Machine (Win Streak / Rolling R)
# - Attack Mode State Machine (Size Scaling)
# - Risk Containment (Drawdown Kill, Cooldowns)
# ============================================================================


# --- Enums for Explicit State ---

class FilterMode(Enum):
    BASE = auto()
    BURST = auto()


class AttackModeState(Enum):
    INACTIVE = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


# --- Immutable Configuration Dataclasses ---

@dataclass(frozen=True)
class FilterConfig:
    sweep_wick_pct: float
    cluster_compression_ratio: float


@dataclass(frozen=True)
class ModeConfig:
    max_drawdown_session_r: float
    max_consecutive_losses: int
    cooldown_trades: int
    min_regime_expectancy: float
    attack_size_mult: float
    hot_streak_size_mult: float
    funding_squeeze_bonus: float


@dataclass(frozen=True)
class SharedParams:
    sweep_thinning_mult: float
    max_bars_since_cluster: int
    prior_extreme_lookback: int
    atr_period: int
    atr_trailing_stop_mult: float
    initial_stop_atr: float
    cooldown_atr: float
    regime_window: int


# --- Mutable State Dataclasses ---

@dataclass
class VirtualTrade:
    direction: str
    entry: float
    stop: float
    initial_stop: float
    atr: float
    size_mult: float


@dataclass
class PendingSweep:
    bar_idx: int
    bias: str
    sweep_row: pd.Series
    atr: float


@dataclass
class DetectorState:
    """Explicit, serializable state container for restart-safety."""
    burst_mode_active: bool = False
    attack_mode_state: AttackModeState = AttackModeState.INACTIVE
    losses_in_attack: int = 0
    wins_in_attack: int = 0
    attack_session_pnl: float = 0.0
    cooldown_counter: int = 0
    last_trade_pnl: float = 0.0
    recent_loss: bool = False
    consecutive_wins: int = 0
    last_cluster_time: Optional[str] = None  # ISO string for serializability

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            'burst_mode_active': self.burst_mode_active,
            'attack_mode_state': self.attack_mode_state.name,
            'losses_in_attack': self.losses_in_attack,
            'wins_in_attack': self.wins_in_attack,
            'attack_session_pnl': self.attack_session_pnl,
            'cooldown_counter': self.cooldown_counter,
            'last_trade_pnl': self.last_trade_pnl,
            'recent_loss': self.recent_loss,
            'consecutive_wins': self.consecutive_wins,
            'last_cluster_time': self.last_cluster_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DetectorState':
        """Deserialize state for restart."""
        return cls(
            burst_mode_active=d.get('burst_mode_active', False),
            attack_mode_state=AttackModeState[d.get('attack_mode_state', 'INACTIVE')],
            losses_in_attack=d.get('losses_in_attack', 0),
            wins_in_attack=d.get('wins_in_attack', 0),
            attack_session_pnl=d.get('attack_session_pnl', 0.0),
            cooldown_counter=d.get('cooldown_counter', 0),
            last_trade_pnl=d.get('last_trade_pnl', 0.0),
            recent_loss=d.get('recent_loss', False),
            consecutive_wins=d.get('consecutive_wins', 0),
            last_cluster_time=d.get('last_cluster_time'),
        )


# --- Pure Functions: Signal Detection ---

def calculate_indicators(df: pd.DataFrame, atr_period: int, prior_extreme_lookback: int) -> pd.DataFrame:
    """Pure function: compute technical indicators. MATCHES BACKTEST EXACTLY."""
    df = df.copy()
    df['range'] = df['high'] - df['low']
    
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    df['atr'] = tr.rolling(atr_period, min_periods=1).mean()
    
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['range'].replace(0, np.nan)
    
    df['roll_max_20'] = df['high'].shift(1).rolling(prior_extreme_lookback, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(prior_extreme_lookback, min_periods=1).min()
    
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
    df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()
    
    return df


def detect_cluster(
    row: pd.Series,
    compression_ratio: float
) -> bool:
    """Pure function: returns True if current bar forms a cluster."""
    tr_cluster = row['tr_roll_cluster']
    tr_prior = row['tr_roll_prior']
    
    if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
        return False
    
    return tr_cluster <= compression_ratio * tr_prior


def check_sweep_eligibility(
    row: pd.Series,
    df: pd.DataFrame,
    last_cluster_time: Optional[str],
    shared_params: SharedParams
) -> bool:
    """Pure function: checks if bar passes shared sweep eligibility filters."""
    # Thinning filter
    if not (row['range'] <= shared_params.sweep_thinning_mult * row['atr']):
        return False
    
    # Volatility pocket required
    if not row['vol_pocket_active']:
        return False
    
    # Check cluster lookback using timestamps
    if last_cluster_time is None:
        return False
        
    cluster_ts = pd.to_datetime(last_cluster_time)
    
    # Count how many bars have passed since the cluster timestamp
    bars_since_cluster = len(df[df['timestamp'] > cluster_ts])
    
    if bars_since_cluster > shared_params.max_bars_since_cluster:
        return False
        
    return True


def detect_sweep_direction(
    row: pd.Series,
    wick_pct_thresh: float
) -> Optional[str]:
    """Pure function: returns 'LONG', 'SHORT', or None based on sweep pattern."""
    # SHORT sweep: upper wick rejection above prior highs
    is_valid_short = (
        row['upper_wick_pct'] > wick_pct_thresh
        and row['high'] > row['roll_max_20'] 
        and row['close'] < row['roll_max_20']
    )
    
    # LONG sweep: lower wick rejection below prior lows
    is_valid_long = (
        row['lower_wick_pct'] > wick_pct_thresh
        and row['low'] < row['roll_min_20'] 
        and row['close'] > row['roll_min_20']
    )
    
    if is_valid_short:
        return 'SHORT'
    elif is_valid_long:
        return 'LONG'
    
    return None


def check_sweep_confirmation(
    sweep_row: pd.Series,
    confirm_row: pd.Series,
    bias: str
) -> bool:
    """
    Pure function: returns True if sweep is confirmed (failed continuation).
    Confirmation = price did NOT continue in sweep direction.
    """
    if bias == 'SHORT':
        # Expecting price to rise (sweep rejection), so failed = close < sweep close
        return confirm_row['close'] < sweep_row['close']
    else:  # LONG
        # Expecting price to fall (sweep rejection), so failed = close > sweep close
        return confirm_row['close'] > sweep_row['close']


# --- Pure Functions: Risk Checks ---

def compute_rolling_r(trade_history: deque, window: int = 10) -> float:
    """Pure function: compute rolling R from trade history."""
    if len(trade_history) < window:
        return 0.0
    return float(np.mean(list(trade_history)[-window:]))


def should_activate_burst_mode(
    current_burst: bool,
    recent_loss: bool,
    consecutive_wins: int,
    rolling_r: float
) -> tuple[bool, Optional[str]]:
    """
    Pure function: determines burst mode transition.
    Returns (new_burst_state, log_reason_or_none).
    """
    if not current_burst:
        # Activation: no recent loss AND (streak >= 3 OR high rolling R)
        if not recent_loss and (consecutive_wins >= 3 or rolling_r > 0.7):
            return True, f"ACTIVATED! Streak: {consecutive_wins}, RollR: {rolling_r:.2f}"
        return False, None
    else:
        # Deactivation: recent loss OR edge fade
        if recent_loss:
            return False, "DEACTIVATED (Recent Loss)"
        if rolling_r < 0.2:
            return False, f"DEACTIVATED (Edge Faded R={rolling_r:.2f})"
        return True, None


def check_attack_kill_conditions(
    state: DetectorState,
    config: ModeConfig
) -> Optional[str]:
    """
    Pure function: returns kill reason if attack mode should be terminated, else None.
    KILL-SWITCH: This function MUST be called before any signal emission.
    """
    if state.attack_mode_state != AttackModeState.ACTIVE:
        return None
    
    if state.losses_in_attack >= config.max_consecutive_losses:
        return "Max Consecutive Losses"
    
    if state.attack_session_pnl <= -config.max_drawdown_session_r:
        return "Session Drawdown Cap"
    
    return None


def should_activate_attack_mode(
    state: DetectorState,
    config: ModeConfig,
    vol_pocket_active: bool,
    rolling_r: float,
    history_len: int
) -> bool:
    """Pure function: determines if attack mode should activate."""
    if state.attack_mode_state != AttackModeState.INACTIVE:
        return False
    
    if state.cooldown_counter > 0:
        return False
    
    if not vol_pocket_active:
        return False
    
    if state.last_trade_pnl < 0 or state.recent_loss:
        return False
    
    # Regime check: only if enough history
    if history_len >= 12:
        if rolling_r < config.min_regime_expectancy:
            return False
    
    return True


def compute_size_multiplier(
    attack_active: bool,
    wins_in_attack: int,
    config: ModeConfig
) -> float:
    """Pure function: compute position size multiplier."""
    if not attack_active:
        return 1.0
    
    if wins_in_attack >= 2:
        return config.hot_streak_size_mult
    
    return config.attack_size_mult


# --- Pure Functions: Execution ---

def compute_initial_stop(
    entry_price: float,
    direction: str,
    atr: float,
    initial_stop_atr: float
) -> float:
    """Pure function: compute initial stop loss price."""
    if direction == 'LONG':
        return entry_price - initial_stop_atr * atr
    else:
        return entry_price + initial_stop_atr * atr


def update_trailing_stop(
    trade: VirtualTrade,
    current_close: float,
    atr_trailing_mult: float
) -> float:
    """Pure function: compute new trailing stop. Returns updated stop value."""
    if trade.direction == 'LONG':
        new_stop = current_close - atr_trailing_mult * trade.atr
        return max(trade.stop, new_stop)
    else:
        new_stop = current_close + atr_trailing_mult * trade.atr
        return min(trade.stop, new_stop)


def check_stop_hit(
    trade: VirtualTrade,
    row: pd.Series
) -> tuple[bool, float]:
    """
    Pure function: check if stop was hit.
    Returns (hit: bool, exit_price: float).
    """
    if trade.direction == 'LONG':
        if row['low'] <= trade.stop:
            return True, trade.stop
    else:
        if row['high'] >= trade.stop:
            return True, trade.stop
    return False, 0.0


def compute_trade_pnl_r(
    trade: VirtualTrade,
    exit_price: float
) -> float:
    """Pure function: compute PnL in R units."""
    if trade.direction == 'LONG':
        return (exit_price - trade.entry) / trade.atr
    else:
        return (trade.entry - exit_price) / trade.atr


# --- Main Class ---

class LiveEventDetectorGem:
    """
    Real-time liquidity event detector (GEM v1.3.0 - Fusion Refactored)
    - Processes 5-minute bars
    - Tracks Cluster State
    - Detects & Confirms Sweeps
    - Manages Burst & Attack Mode State Machines
    """

    def __init__(
        self,
        symbol: str,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        enable_telegram: bool = True,
        telegram_bot=None
    ):
        self.symbol = symbol

        # --- Immutable Configuration ---
        self.BASE_FILTERS = FilterConfig(
            sweep_wick_pct=0.45,
            cluster_compression_ratio=0.6
        )

        self.BURST_FILTERS = FilterConfig(
            sweep_wick_pct=0.35,
            cluster_compression_ratio=0.7
        )

        self.BURST_CONFIG = ModeConfig(
            max_drawdown_session_r=10.0,
            max_consecutive_losses=10,
            cooldown_trades=0,
            min_regime_expectancy=-5.0,
            attack_size_mult=2.0,
            hot_streak_size_mult=2.5,
            funding_squeeze_bonus=0.5
        )

        self.BASE_CONFIG = ModeConfig(
            max_drawdown_session_r=2.0,
            max_consecutive_losses=3,
            cooldown_trades=8,
            min_regime_expectancy=-0.3,
            attack_size_mult=1.5,
            hot_streak_size_mult=1.8,
            funding_squeeze_bonus=0.2
        )

        self.SHARED_PARAMS = SharedParams(
            sweep_thinning_mult=1.2,
            max_bars_since_cluster=20,
            prior_extreme_lookback=20,
            atr_period=20,
            atr_trailing_stop_mult=1.8,
            initial_stop_atr=0.95,
            cooldown_atr=1.2,
            regime_window=10
        )

        # --- Mutable State ---
        self.buffer: deque = deque(maxlen=100)
        self.pending_sweep: Optional[PendingSweep] = None
        self.state = DetectorState()
        self.virtual_trades: List[VirtualTrade] = []
        self.trade_history: deque = deque(maxlen=20)

        # --- External Dependencies ---
        self.event_callback = event_callback or self._default_callback
        self.telegram = telegram_bot if telegram_bot else (TelegramBot() if enable_telegram else None)
        self.lock = threading.Lock()
        self._setup_logging()

        self.logger.info(f"GEM v1.3.0 (Option F Refactored) Initialized for {symbol}")

    def _setup_logging(self) -> None:
        """Side effect: configures file logging."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f'GEM_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, 'gem_detector.log'))
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            self.logger.addHandler(handler)
            
            # Add StreamHandler for visibility in PM2/System logs
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            self.logger.addHandler(stream_handler)

    # --- Public API ---

    def on_bar(self, bar: Dict[str, Any]) -> None:
        """
        Ingest new bar. Thread-safe entry point.
        Early return: if bar is missing required fields.
        """
        with self.lock:
            try:
                required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(k in bar for k in required):
                    return  # Intentional early return: invalid bar data
                
                self.buffer.append(bar)
                if len(self.buffer) >= 25:
                    self._process_bar()
                else:
                    if len(self.buffer) % 5 == 0:
                        self.logger.info(f"Buffer warming up: {len(self.buffer)}/25 bars...")

            except Exception as e:
                self.logger.error(f"Error processing bar: {e}", exc_info=True)

    def get_state(self) -> Dict[str, Any]:
        """Returns serializable state for persistence."""
        with self.lock:
            # Convert buffer timestamps to strings for JSON
            buffer_json = []
            for b in list(self.buffer):
                b_copy = b.copy()
                if isinstance(b_copy['timestamp'], (pd.Timestamp, datetime)):
                    b_copy['timestamp'] = b_copy['timestamp'].isoformat()
                buffer_json.append(b_copy)
                
            # Convert trade history objects to dicts
            history_json = []
            for t in self.trade_history:
                if hasattr(t, '__dict__'):
                    history_json.append(t.__dict__)
                else:
                    history_json.append(t)

            # Convert any numpy/pandas types to Python native types for JSON
            def convert_to_serializable(obj):
                if obj is None:
                    return None
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(v) for v in obj]
                return obj
            
            state_dict = convert_to_serializable(self.state.to_dict())
            
            return {
                "state": state_dict,
                "buffer": convert_to_serializable(buffer_json),
                "pending_sweep": convert_to_serializable(self.pending_sweep.__dict__) if hasattr(self.pending_sweep, '__dict__') else self.pending_sweep,
                "trade_history": convert_to_serializable(history_json)
            }

    def restore_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore state and historical buffer from serialized dict."""
        with self.lock:
            # 1. Restore basic flags
            if 'state' in state_dict:
                self.state = DetectorState.from_dict(state_dict['state'])
            
            # 2. Restore buffer data
            if 'buffer' in state_dict:
                restored_buffer = deque(maxlen=self.buffer.maxlen)
                for b in state_dict['buffer']:
                    if isinstance(b.get('timestamp'), str):
                        b['timestamp'] = pd.to_datetime(b['timestamp'])
                    restored_buffer.append(b)
                self.buffer = restored_buffer
            
            # 3. Restore trade history
            if 'trade_history' in state_dict:
                self.trade_history = []
                for t in state_dict['trade_history']:
                    if isinstance(t, dict):
                        self.trade_history.append(VirtualTrade(**t))
                    else:
                        self.trade_history.append(t)
            
            self.pending_sweep = state_dict.get('pending_sweep')
            
            self.logger.info(f"✅ Restored {self.symbol} with {len(self.buffer)} bars in buffer.")

    # --- Main Processing Pipeline ---

    def _process_bar(self) -> None:
        """
        Main processing loop. Phases execute in strict order:
        1. Indicators
        2. Virtual trade management (updates PnL state)
        3. Burst mode state transition
        4. Attack mode state transition (includes kill-switch)
        5. Cluster detection
        6. Pending sweep confirmation
        7. New sweep detection
        """
        df = pd.DataFrame(list(self.buffer))
        current_idx = len(df) - 1
        row = df.iloc[-1]

        # Phase 1: Calculate indicators (pure function)
        df = calculate_indicators(
            df,
            self.SHARED_PARAMS.atr_period,
            self.SHARED_PARAMS.prior_extreme_lookback
        )
        row = df.iloc[-1]  # Re-fetch row after indicator computation

        # Phase 2: Manage virtual trades (side effect: updates state)
        self._manage_virtual_trades(row)

        # Phase 3: Update burst mode (side effect: updates state.burst_mode_active)
        self._update_burst_state()

        # Phase 4: Update attack mode (side effect: updates state, includes KILL-SWITCH)
        self._manage_attack_mode_state(row)

        # Phase 5: Update cluster state (side effect: updates state.last_cluster_end_idx)
        self._update_cluster_state(df, current_idx)

        # Phase 6: Process pending sweep confirmation
        if self.pending_sweep is not None:
            self._check_confirmation(df, self.pending_sweep)
            self.pending_sweep = None

        # Phase 7: Detect new sweep
        self._detect_sweep(df, current_idx)

    # --- State Machine Updates ---

    def _update_burst_state(self) -> None:
        """Side effect: updates self.state.burst_mode_active based on trade history."""
        rolling_r = compute_rolling_r(self.trade_history, window=10)

        new_burst, log_reason = should_activate_burst_mode(
            current_burst=self.state.burst_mode_active,
            recent_loss=self.state.recent_loss,
            consecutive_wins=self.state.consecutive_wins,
            rolling_r=rolling_r
        )

        if new_burst != self.state.burst_mode_active:
            self.state.burst_mode_active = new_burst
            if log_reason:
                self.logger.info(f"[BURST] {log_reason}")

    def _manage_attack_mode_state(self, row: pd.Series) -> None:
        """
        Side effect: updates attack mode state.
        KILL-SWITCH CHECK: executed first, cannot be bypassed.
        """
        config = self._get_mode_config()

        # KILL-SWITCH: Check kill conditions FIRST (mandatory, cannot bypass)
        kill_reason = check_attack_kill_conditions(self.state, config)
        if kill_reason:
            self._execute_attack_kill(kill_reason, config)
            return  # Intentional early return: kill executed

        # Handle cooldown decrement
        if self.state.attack_mode_state == AttackModeState.COOLDOWN:
            if self.state.cooldown_counter > 0:
                pass  # Cooldown managed at signal emission
            else:
                self.state.attack_mode_state = AttackModeState.INACTIVE

        # Check activation conditions
        if self.state.attack_mode_state == AttackModeState.INACTIVE:
            rolling_r = compute_rolling_r(self.trade_history, window=10)
            should_activate = should_activate_attack_mode(
                state=self.state,
                config=config,
                vol_pocket_active=row['vol_pocket_active'],
                rolling_r=rolling_r,
                history_len=len(self.trade_history)
            )

            if should_activate:
                self._activate_attack_mode()

    def _execute_attack_kill(self, reason: str, config: ModeConfig) -> None:
        """Side effect: executes attack mode kill and enters cooldown."""
        self.state.attack_mode_state = AttackModeState.COOLDOWN
        self.state.cooldown_counter = config.cooldown_trades
        self.state.wins_in_attack = 0

        # Cascade: kill burst mode if hard stop hit
        if self.state.burst_mode_active:
            self.state.burst_mode_active = False

        self.logger.warning(f"[ATTACK KILL] {reason}. Cooldown: {self.state.cooldown_counter}")

    def _activate_attack_mode(self) -> None:
        """Side effect: activates attack mode with fresh counters."""
        self.state.attack_mode_state = AttackModeState.ACTIVE
        self.state.losses_in_attack = 0
        self.state.wins_in_attack = 0
        self.state.attack_session_pnl = 0.0
        self.logger.info(f"[ATTACK START] Scaling Enabled (Burst={self.state.burst_mode_active})")

    # --- Detection ---

    def _update_cluster_state(self, df: pd.DataFrame, current_idx: int) -> None:
        """Side effect: updates state.last_cluster_time if cluster detected."""
        row = df.iloc[current_idx]
        filters = self._get_filter_config()

        if detect_cluster(row, filters.cluster_compression_ratio):
            ts = row['timestamp']
            if isinstance(ts, (pd.Timestamp, datetime)):
                self.state.last_cluster_time = ts.isoformat()
            else:
                self.state.last_cluster_time = str(ts)
            self.logger.info(f"CLUSTER DETECTED at {self.state.last_cluster_time} (Ratio: {row['tr_roll_cluster']/row['tr_roll_prior']:.2f})")

    def _detect_sweep(self, df: pd.DataFrame, current_idx: int) -> None:
        """Side effect: sets self.pending_sweep if sweep pattern detected."""
        row = df.iloc[current_idx]

        # Check eligibility (pure function)
        if not check_sweep_eligibility(
            row, df,
            self.state.last_cluster_time,
            self.SHARED_PARAMS
        ):
            return  # Intentional early return: not eligible

        # Check direction (pure function)
        filters = self._get_filter_config()
        direction = detect_sweep_direction(row, filters.sweep_wick_pct)

        if direction:
            self.pending_sweep = PendingSweep(
                bar_idx=current_idx,
                bias=direction,
                sweep_row=row,
                atr=row['atr']
            )
            self.logger.info(f"ARMED SWEEP ({direction}) - Burst: {self.state.burst_mode_active}")

    def _check_confirmation(self, df: pd.DataFrame, pending: PendingSweep) -> None:
        """Side effect: emits signal if sweep confirmation passes."""
        current_idx = len(df) - 1
        confirm_row = df.iloc[current_idx]

        if check_sweep_confirmation(pending.sweep_row, confirm_row, pending.bias):
            self._emit_signal(confirm_row, pending.bias, pending.atr)
        else:
            self.logger.info(f"Sweep {pending.bar_idx} failed confirmation.")

    # --- Signal Emission ---

    def _emit_signal(self, bar: pd.Series, direction: str, atr: float) -> None:
        """
        Side effect: emits trading signal, creates virtual trade, sends notifications.
        """
        signal_timestamp = datetime.utcnow()  # Signal generation time, not bar time
        entry_price = bar['close']

        # Compute sizing (pure function)
        config = self._get_mode_config()
        attack_active = self.state.attack_mode_state == AttackModeState.ACTIVE
        size_mult = compute_size_multiplier(
            attack_active,
            self.state.wins_in_attack,
            config
        )

        # Decrement cooldown if active
        if self.state.cooldown_counter > 0:
            self.state.cooldown_counter -= 1

        self.logger.info(
            f"🚀 SIGNAL: {direction} @ {entry_price}, Size: {size_mult}x "
            f"(Burst:{self.state.burst_mode_active})"
        )

        # Compute stop (pure function)
        stop = compute_initial_stop(
            entry_price, direction, atr,
            self.SHARED_PARAMS.initial_stop_atr
        )

        # Track virtual trade
        self.virtual_trades.append(VirtualTrade(
            direction=direction,
            entry=entry_price,
            stop=stop,
            initial_stop=stop,
            atr=atr,
            size_mult=size_mult
        ))

        # Telegram notification (side effect)
        self._send_telegram_alert(direction, entry_price, stop, atr, signal_timestamp)

        # Build and emit event
        event_data = self._build_event_data(
            bar, direction, entry_price, stop, atr, size_mult, signal_timestamp
        )
        self._dispatch_event(event_data)

    def _send_telegram_alert(
        self,
        direction: str,
        entry_price: float,
        stop: float,
        atr: float,
        timestamp: Any
    ) -> None:
        """Side effect: sends Telegram notification if configured. Thread-safe."""
        if not self.telegram:
            return
        
        # Schedule on main event loop using run_coroutine_threadsafe
        async def send_alert():
            try:
                await self.telegram.send_entry_alert(
                    pair=self.symbol,
                    direction=direction,
                    entry_price=entry_price,
                    event_type=f"GEM_FUSION (Burst={self.state.burst_mode_active})",
                    stop_loss=stop,
                    take_profit=0,
                    atr=atr,
                    timestamp=timestamp,
                    trailing_stop_atr=self.SHARED_PARAMS.atr_trailing_stop_mult
                )
            except Exception as e:
                self.logger.error(f"Telegram alert error: {e}", exc_info=True)
        
        try:
            # Try to get the running loop and schedule the coroutine
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(send_alert(), loop)
        except RuntimeError:
            # No loop running, use threading with asyncio.run
            def send_async():
                try:
                    asyncio.run(send_alert())
                except Exception as e:
                    self.logger.error(f"Telegram alert error: {e}", exc_info=True)
            threading.Thread(target=send_async, daemon=True).start()

    def _build_event_data(
        self,
        bar: pd.Series,
        direction: str,
        entry_price: float,
        stop: float,
        atr: float,
        size_mult: float,
        timestamp: Any
    ) -> Dict[str, Any]:
        """Pure function: builds event data dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'event_type': 'GEM_SWEEP',
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop,
            'take_profit': 0,
            'atr': atr,
            'size_multiplier': size_mult,
            'volume': bar['volume'],
            'bar': bar.to_dict(),
            'meta_attack_mode': self.state.attack_mode_state == AttackModeState.ACTIVE,
            'meta_burst_mode': self.state.burst_mode_active,
            'meta_filters': 'BURST' if self.state.burst_mode_active else 'BASE'
        }

    def _dispatch_event(self, event_data: Dict[str, Any]) -> None:
        """Side effect: dispatches event to callback. Thread-safe for async callbacks."""
        if not self.event_callback:
            return
        
        if asyncio.iscoroutinefunction(self.event_callback):
            # For async callbacks, use asyncio.run() in a new thread
            def run_async():
                try:
                    asyncio.run(self.event_callback(event_data))
                except Exception as e:
                    self.logger.error(f"Async callback error: {e}", exc_info=True)
            threading.Thread(target=run_async, daemon=True).start()
        else:
            threading.Thread(
                target=self.event_callback,
                args=(event_data,),
                daemon=True
            ).start()

    # --- Virtual Trade Management ---

    def _manage_virtual_trades(self, row: pd.Series) -> None:
        """Side effect: updates virtual trades, closes stopped trades, updates PnL state."""
        active_trades: List[VirtualTrade] = []

        for trade in self.virtual_trades:
            hit, exit_price = check_stop_hit(trade, row)

            if hit:
                pnl_r = compute_trade_pnl_r(trade, exit_price)
                final_r = pnl_r * trade.size_mult
                self._update_state_outcome(final_r, trade, exit_price)
            else:
                # Update trailing stop
                old_stop = trade.stop
                trade.stop = update_trailing_stop(
                    trade,
                    row['close'],
                    self.SHARED_PARAMS.atr_trailing_stop_mult
                )
                # Send TSL notification if stop moved
                if trade.stop != old_stop and self.telegram:
                    async def send_tsl():
                        try:
                            await self.telegram.send_tsl_update(
                                pair=self.symbol,
                                direction=trade.direction,
                                new_stop=trade.stop,
                                entry_price=trade.entry
                            )
                        except Exception as e:
                            self.logger.error(f"TSL alert failed: {e}")
                    
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(send_tsl(), loop)
                    except RuntimeError:
                        def send_async():
                            try:
                                asyncio.run(send_tsl())
                            except Exception as e:
                                self.logger.error(f"TSL alert failed: {e}")
                        threading.Thread(target=send_async, daemon=True).start()
                active_trades.append(trade)

        self.virtual_trades = active_trades

    def _update_state_outcome(self, pnl_r: float, trade: VirtualTrade = None, exit_price: float = None) -> None:
        """
        Side effect: updates all PnL-related state.
        Called upon trade closure (virtual or real).
        """
        self.logger.info(f"Trade Result: {pnl_r:.2f}R")
        
        # Send exit notification if trade details available
        if trade and exit_price and self.telegram:
            async def send_exit():
                try:
                    await self.telegram.send_exit_alert(
                        pair=self.symbol,
                        direction=trade.direction,
                        exit_price=exit_price,
                        pnl_r=pnl_r,
                        timestamp=datetime.now()
                    )
                except Exception as e:
                    self.logger.error(f"Exit alert failed: {e}")
            
            try:
                # Try to get the running loop and schedule the coroutine
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(send_exit(), loop)
            except RuntimeError:
                # No loop running, use threading with asyncio.run
                def send_async():
                    try:
                        asyncio.run(send_exit())
                    except Exception as e:
                        self.logger.error(f"Exit alert failed: {e}")
                threading.Thread(target=send_async, daemon=True).start()
        
        # Log metrics to CSV
        if trade and exit_price:
            try:
                import csv
                import os
                os.makedirs('analysis', exist_ok=True)
                metrics_file = 'analysis/trade_metrics.csv'
                file_exists = os.path.exists(metrics_file)
                
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'symbol', 'direction', 'entry_price', 'exit_price', 'pnl_r', 'atr'])
                    writer.writerow([
                        datetime.now().isoformat(),
                        self.symbol,
                        trade.direction,
                        trade.entry,
                        exit_price,
                        round(pnl_r, 2),
                        trade.atr
                    ])
                self.logger.info(f"Metrics logged to {metrics_file}")
            except Exception as e:
                self.logger.error(f"Metrics logging failed: {e}")
        
        self.state.last_trade_pnl = pnl_r
        self.state.recent_loss = (pnl_r < 0)
        self.trade_history.append(pnl_r)

        # Streak update
        if pnl_r > 0:
            self.state.consecutive_wins += 1
        else:
            self.state.consecutive_wins = 0

        # Attack session accumulation
        if self.state.attack_mode_state == AttackModeState.ACTIVE:
            self.state.attack_session_pnl += pnl_r
            if pnl_r < 0:
                self.state.losses_in_attack += 1
                self.state.wins_in_attack = 0
            else:
                self.state.wins_in_attack += 1

    # --- Helpers ---

    def _get_filter_config(self) -> FilterConfig:
        """Returns active filter config based on burst mode."""
        return self.BURST_FILTERS if self.state.burst_mode_active else self.BASE_FILTERS

    def _get_mode_config(self) -> ModeConfig:
        """Returns active mode config based on burst mode."""
        return self.BURST_CONFIG if self.state.burst_mode_active else self.BASE_CONFIG

    def _default_callback(self, data: Dict[str, Any]) -> None:
        """Default callback for signal events."""
        print(f"[GEM] {data['direction']} Signal at {data['timestamp']}")

    # --- External State Update (for real trade results) ---

    def update_state_outcome(self, pnl_r: float) -> None:
        """Public API: called upon trade closure (virtual or real)."""
        with self.lock:
            self._update_state_outcome(pnl_r)
