# risk_manager.py
"""
Phase 2 Risk Engine: The "No" Layer.
Tuned based on Backtest 2026-01-17:
- REMOVED: Session Scaling (Proven to reduce Expectancy).
- KEPT: Correlation Gating (Systemic Risk protection).
- MODIFIED: Daily Limit loose (-10R) for Catastrophic Protection only.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any

class RiskManager:
    def __init__(self, daily_loss_limit_r: float = -10.0, max_active_correlated: int = 1):
        self._setup_logging()
        
        # Configuration
        self.DAILY_LOSS_LIMIT_R = daily_loss_limit_r
        self.MAX_ACTIVE_CORRELATED = max_active_correlated
        
        # State
        self.daily_pnl_r = 0.0
        self.current_date = datetime.utcnow().date()
        
        # Correlation Buckets
        # Map symbol -> Bucket
        self.asset_buckets = {
            'SOLUSDT': 'L1',
            'ETHUSDT': 'L1',
            'AVAXUSDT': 'L1',
            'SUIUSDT': 'L1',
            'DOGEUSDT': 'MEME',
            'PEPEUSDT': 'MEME'
        }
        
        # Track active positions per bucket
        # { 'L1': 0, 'MEME': 0 }
        self.active_buckets = {k: 0 for k in set(self.asset_buckets.values())}
        
        self.logger.info("🛡️ Risk Manager Initialized (Tuned v1.1)")
        self.logger.info(f"Circuit Breaker: {self.DAILY_LOSS_LIMIT_R}R | Max Correlated: {self.MAX_ACTIVE_CORRELATED}")

    def _setup_logging(self):
        self.logger = logging.getLogger('RiskManager')
        self.logger.setLevel(logging.INFO)

    def _reset_daily_if_new_day(self):
        today = datetime.utcnow().date()
        if today > self.current_date:
            self.daily_pnl_r = 0.0
            self.current_date = today
            self.logger.info(f"New Day Detected ({today}). Daily PnL Reset.")

    def update_pnl(self, pnl_r: float):
        """Update Daily PnL state from Validator/Executor feedback"""
        self._reset_daily_if_new_day()
        self.daily_pnl_r += pnl_r
        self.logger.info(f"Daily PnL Update: {self.daily_pnl_r:.2f}R")

    def register_entry(self, symbol: str):
        """Call on confirmed entry to update correlation state"""
        bucket = self.asset_buckets.get(symbol, 'OTHER')
        if bucket not in self.active_buckets: self.active_buckets[bucket] = 0
        self.active_buckets[bucket] += 1
        # self.logger.info(f"Correlation Update: {bucket} count -> {self.active_buckets[bucket]}")

    def register_exit(self, symbol: str):
        """Call on confirmed exit to update correlation state"""
        bucket = self.asset_buckets.get(symbol, 'OTHER')
        if bucket in self.active_buckets and self.active_buckets[bucket] > 0:
            self.active_buckets[bucket] -= 1
        # self.logger.info(f"Correlation Update: {bucket} count -> {self.active_buckets[bucket]}")

    def validate_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The Main Gatekeeper.
        Returns: Signal (passed through) OR None (Rejection).
        """
        self._reset_daily_if_new_day()
        symbol = signal['symbol']
        
        # 1. CIRCUIT BREAKER: Daily Loss
        if self.daily_pnl_r <= self.DAILY_LOSS_LIMIT_R:
            self.logger.warning(f"🛑 REJECTED {symbol}: Daily Circuit Breaker Hit ({self.daily_pnl_r}R)")
            return None
            
        # 2. CORRELATION GUARD
        bucket = self.asset_buckets.get(symbol, 'OTHER')
        current_active = self.active_buckets.get(bucket, 0)
        if current_active >= self.MAX_ACTIVE_CORRELATED:
            self.logger.warning(f"🛑 REJECTED {symbol}: Correlation Bucket '{bucket}' Full ({current_active}/{self.MAX_ACTIVE_CORRELATED})")
            return None
        
        # 3. SESSION SCALING (REMOVED based on 2026-01-17 Backtest)
        # Original logic removed. Asia session proved profitable.
        
        # Return original signal
        return signal
