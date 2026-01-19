# live_event_detector_gem.py

import pandas as pd
import numpy as np
from collections import deque
import logging
from datetime import datetime
import threading
from typing import Dict, Any, Optional, Callable
import asyncio
import os
from telegram_bot import TelegramBot

# ============================================================================
# LIVE EVENT DETECTOR GEM (v1.1.0)
# ============================================================================
# Features:
# - Sweep + Cluster + Thinning
# - Sticky Attack Mode (Performance-based)
# - Aggressive Sizing (1.5x)
# - Risk Containment (Drawdown Kill, Cooldowns)
# ============================================================================

class LiveEventDetectorGem:
    """
    Real-time liquidity event detector (GEM Strategy)
    - Processes 5-minute bars
    - Tracks Cluster State
    - Detects & Confirms Sweeps
    - Manages Attack Mode State Machine
    """
    
    def __init__(self, symbol: str, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 enable_telegram: bool = True, telegram_bot=None):
        self.symbol = symbol
        
        # PARAMETERS (Synced with v1.1.0 Backtest)
        self.PARAMS = {
            'SWEEP_THINNING_MULT': 1.2,
            'SWEEP_WICK_PCT': 0.35,
            'CLUSTER_LEN': 5,
            'COOLDOWN_ATR': 1.5,
            'MAX_BARS_SINCE_CLUSTER': 20,
            'PRIOR_EXTREME_LOOKBACK': 20,
            'ATR_PERIOD': 20,
            'VOL_MEDIAN_PERIOD': 50,
            'ATR_TRAILING_STOP_MULT': 1.8,
            'INITIAL_STOP_ATR': 1.0,
            
            # CONTAINMENT
            'MAX_DRAWDOWN_SESSION_R': 2.0,
            'MAX_CONSECUTIVE_LOSSES': 2,
            'COOLDOWN_TRADES': 5,
            'REGIME_WINDOW': 10
        }
        
        # Buffers
        self.buffer = deque(maxlen=100)
        self.pending_sweep = None
        self.last_cluster_end_idx = -999
        
        # ATTACK MODE STATE
        self.attack_mode_active = False
        self.losses_in_attack = 0
        self.attack_session_pnl = 0.0
        self.cooldown_counter = 0      # Trades remaining in cooldown
        self.last_trade_pnl = 0.0
        self.recent_loss = False
        
    # Virtual PnL Tracking (for self-containment)
        # We track signals as "Virtual Trades" to calculate theoretical PnL
        # and drive the Attack Mode state machine autonomously.
        self.virtual_trades = [] # List of dicts: {direction, entry, stop, size_mult}
        self.trade_history = deque(maxlen=20) 
        
        self.event_callback = event_callback or self._default_callback
        
        # Telegram
        self.telegram = telegram_bot if telegram_bot else (TelegramBot() if enable_telegram else None)
        
        # Threading/Logging
        self.lock = threading.Lock()
        self._setup_logging()
        
        self.logger.info("LiveEventDetectorGem v1.1.0 initialized (Aggressive + Contained + VirtualTracking)")

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f'LiveEventDetectorGem_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, 'event_detector_gem.log'))
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            self.logger.addHandler(handler)

    def on_5min_bar(self, bar: Dict[str, Any]):
        """Ingest new 5-min bar"""
        with self.lock:
            try:
                required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(k in bar for k in required): return

                self.buffer.append(bar)
                if len(self.buffer) > 50:
                    self._process_bar()
                    
            except Exception as e:
                self.logger.error(f"Error processing bar: {e}", exc_info=True)

    def _process_bar(self):
        """Main Logic Loop"""
        df = pd.DataFrame(list(self.buffer))
        current_idx = len(df) - 1
        row = df.iloc[-1]
        
        # 1. Update Indicators
        self._calculate_indicators(df)
        
        # 1.5 Update Virtual Trades (Simulation of PnL)
        self._manage_virtual_trades(row)
        
        # 2. Update Attack Mode State (Driven by Virtual PnL)
        self._manage_attack_mode_state(row)
        
        # 3. Detect Clusters
        self._update_cluster_state(df, current_idx)
        
        # 4. Process Pending Sweep
        if self.pending_sweep:
            self._check_confirmation(df, self.pending_sweep)
            self.pending_sweep = None
            
        # 5. Detect New Sweep
        self._detect_sweep(df, current_idx)

    def _manage_virtual_trades(self, row: pd.Series):
        """
        Simulate trade outcomes (Trailing Stop) for active signals 
        to generate PnL feedback for state validation.
        """
        active = []
        for trade in self.virtual_trades:
            # Check Stop Loss
            pnl_r = 0.0
            closed = False
            
            if trade['direction'] == 'LONG':
                if row['low'] <= trade['stop']:
                    # Stopped out
                    exit_price = trade['stop']
                    risk = trade['entry'] - trade['initial_stop'] # distance
                    # R = (Exit - Entry) / Risk_Distance (approx, simplied)
                    # Actually standard R calculation: (Exit - Entry) / (Entry - InitStop)
                    # If stops are dynamic, we use ATR basis.
                    # Simple: (Exit - Entry) / trade['atr']
                    pnl_r = (exit_price - trade['entry']) / trade['atr']
                    closed = True
                else:
                    # Update Trailing Stop
                    new_stop = row['close'] - self.PARAMS['ATR_TRAILING_STOP_MULT'] * row['atr']
                    if new_stop > trade['stop']:
                        trade['stop'] = new_stop
                        # self.logger.info(f"Trailing Stop Update (LONG): {new_stop:.2f}")
            else: # SHORT
                if row['high'] >= trade['stop']:
                    # Stopped out
                    exit_price = trade['stop']
                    pnl_r = (trade['entry'] - exit_price) / trade['atr']
                    closed = True
                else:
                    # Update Trailing Stop
                    new_stop = row['close'] + self.PARAMS['ATR_TRAILING_STOP_MULT'] * row['atr']
                    if new_stop < trade['stop']:
                        trade['stop'] = new_stop
                        # self.logger.info(f"Trailing Stop Update (SHORT): {new_stop:.2f}")
            
            if closed:
                # Apply Size Mult
                final_r = pnl_r * trade['size_mult']
                self.update_trade_result(final_r) # Update State Machine
            else:
                active.append(trade)
                
        self.virtual_trades = active

    def _calculate_indicators(self, df: pd.DataFrame):
        df['range'] = df['high'] - df['low']
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.PARAMS['ATR_PERIOD'], min_periods=1).mean()
        
        # Volatility Pocket
        df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
        df['atr_slope'] = df['atr'].diff(5)
        # Note: We don't save 'vol_pocket_active' column, we compute on fly or use column
        df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
        
        # Clusters
        df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
        df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()
        
        # Wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df['range']
        df['lower_wick_pct'] = df['lower_wick'] / df['range']
        
        # Extremes
        df['roll_max_20'] = df['high'].shift(1).rolling(self.PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).max()
        df['roll_min_20'] = df['low'].shift(1).rolling(self.PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).min()

    def _manage_attack_mode_state(self, row: pd.Series):
        """Update Attack Mode based on Containment Logic"""
        
        # DAMAGE-BASED INVALIDATION
        kill_reason = None
        if self.attack_mode_active:
            if self.losses_in_attack >= self.PARAMS['MAX_CONSECUTIVE_LOSSES']:
                kill_reason = "Max Consecutive Losses"
            elif self.attack_session_pnl <= -self.PARAMS['MAX_DRAWDOWN_SESSION_R']:
                kill_reason = "Max Session Drawdown"
            
            if kill_reason:
                self.attack_mode_active = False
                self.cooldown_counter = self.PARAMS['COOLDOWN_TRADES']
                self.logger.warning(f"[ATTACK KILL] Mode OFF. Reason: {kill_reason}. Cooldown: {self.cooldown_counter} trades.")
                return

        # ACTIVATION LOGIC
        if not self.attack_mode_active:
            if self.cooldown_counter > 0:
                pass # Still cooling down
            else:
                is_pocket = row['vol_pocket_active']
                if (is_pocket and self.last_trade_pnl >= 0 and not self.recent_loss):
                    # Regime Governor (Rolling Expectancy)
                    regime_ok = True
                    if len(self.trade_history) >= self.PARAMS['REGIME_WINDOW']:
                        rolling_r = np.mean(list(self.trade_history)[-self.PARAMS['REGIME_WINDOW']:])
                        if rolling_r < 0:
                            regime_ok = False
                            # self.logger.info(f"Regime Block: Rolling R {rolling_r:.2f} < 0")
                    
                    if regime_ok:
                        self.attack_mode_active = True
                        self.losses_in_attack = 0
                        self.attack_session_pnl = 0.0
                        self.logger.info("[ATTACK START] Mode ON! Aggressive Sizing Enabled.")

    def _update_cluster_state(self, df: pd.DataFrame, current_idx: int):
        row = df.iloc[current_idx]
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0: return
        
        if tr_cluster <= 0.7 * tr_prior:
            self.last_cluster_end_idx = current_idx

    def _detect_sweep(self, df: pd.DataFrame, current_idx: int):
        row = df.iloc[current_idx]
        
        # 1. Eligibility (Thinning + Pocket)
        if not (row['range'] <= self.PARAMS['SWEEP_THINNING_MULT'] * row['atr']): return
        if not ((row['atr'] > row['atr_med_20']) and (row['atr_slope'] > 0)): return
        
        # 2. Cluster Recency
        if (current_idx - self.last_cluster_end_idx) > self.PARAMS['MAX_BARS_SINCE_CLUSTER']: return
        
        # 3. Extremes
        active_bias = None
        if (row['upper_wick_pct'] > self.PARAMS['SWEEP_WICK_PCT'] and 
            row['high'] > row['roll_max_20'] and row['close'] < row['roll_max_20']):
            active_bias = 'SHORT'
        elif (row['lower_wick_pct'] > self.PARAMS['SWEEP_WICK_PCT'] and
              row['low'] < row['roll_min_20'] and row['close'] > row['roll_min_20']):
            active_bias = 'LONG'
            
        if active_bias:
            self.pending_sweep = {
                'bar_idx': current_idx,
                'bias': active_bias,
                'sweep_row': row,
                'atr': row['atr']
            }
            self.logger.info(f"ARMED SWEEP: {active_bias} at {row['timestamp']}")

    def _check_confirmation(self, df: pd.DataFrame, pending: Dict):
        current_idx = len(df) - 1
        confirm = df.iloc[current_idx]
        sweep = pending['sweep_row']
        bias = pending['bias']
        
        # Failed Continuation Check
        failed = False
        if bias == 'SHORT': failed = confirm['close'] < sweep['close']
        else: failed = confirm['close'] > sweep['close']
            
        if failed:
            self._emit_signal(confirm, bias, sweep['atr'])
        else:
            self.logger.info(f"Sweep {pending['bar_idx']} failed confirmation.")

    def _emit_signal(self, bar: pd.Series, direction: str, atr: float):
        timestamp = bar['timestamp']
        entry_price = bar['close']
        
        size_mult = 1.0
        if self.attack_mode_active:
            size_mult = 1.5
            self.logger.info("[AGGRESSIVE] Signal Sizing: 1.5x")
            
        if self.cooldown_counter > 0:
             self.cooldown_counter -= 1 # Decrement cooldown on trade execution attempt
        
        self.logger.info(f"🚀 SIGNAL: {direction} @ {entry_price}, Size: {size_mult}x")
        
        if direction == 'LONG': stop = entry_price - self.PARAMS['INITIAL_STOP_ATR'] * atr
        else: stop = entry_price + self.PARAMS['INITIAL_STOP_ATR'] * atr
        
        # Track Virtual Trade
        self.virtual_trades.append({
            'direction': direction,
            'entry': entry_price,
            'stop': stop,
            'initial_stop': stop,
            'atr': atr,
            'size_mult': size_mult
        })
        
        if self.telegram:
            asyncio.create_task(self.telegram.send_entry_alert(
                pair=self.symbol, direction=direction, entry_price=entry_price, 
                event_type="GEM_SWEEP", stop_loss=stop, take_profit=0, atr=atr, 
                timestamp=timestamp, trailing_stop_atr=self.PARAMS['ATR_TRAILING_STOP_MULT']
            ))

        event_data = {
            'symbol': self.symbol, 'timestamp': timestamp, 'event_type': 'GEM_SWEEP',
            'direction': direction, 'entry_price': entry_price, 'stop_loss': stop, 'take_profit': 0,
            'atr': atr, 'size_multiplier': size_mult, 'volume': bar['volume'], 'bar': bar.to_dict(),
            # TELEMETRY ENHANCEMENTS
            'meta_attack_mode': self.attack_mode_active,
            'meta_bar_range': bar['range'],
            'meta_bar_close_time': bar['timestamp'] # Redundant but explicit
        }
        
        if self.event_callback and asyncio.iscoroutinefunction(self.event_callback):
            asyncio.run_coroutine_threadsafe(self.event_callback(event_data), asyncio.get_event_loop())
        else:
            threading.Thread(target=self.event_callback, args=(event_data,), daemon=True).start()

    def update_trade_result(self, pnl_r: float):
        """
        CRITICAL: This method must be called by the Execution Engine / Orchestrator
        when a trade closes, to update the Attack Mode state machine.
        """
        with self.lock:
            self.logger.info(f"Trade Closed. PnL: {pnl_r:.2f}R")
            self.last_trade_pnl = pnl_r
            self.recent_loss = (pnl_r < 0)
            self.trade_history.append(pnl_r)
            
            if self.attack_mode_active:
                self.attack_session_pnl += pnl_r
                if pnl_r < 0:
                    self.losses_in_attack += 1
                else:
                    self.losses_in_attack = 0 # Reset on win? Or keep cumulative? 
                    # Backtest logic: losses_in_attack resets on win?
                    # "Track consecutive losing trades" -> implies reset on win. Yes.
                    pass

    def _default_callback(self, data):
        print(f"[GEM] {data['direction']} Signal at {data['timestamp']}")

# Helper for integration
def integrate_with_feed_handler_gem(feed_handler, detector):
    def on_5min_resampled(bar):
        detector.on_5min_bar(bar)
    return on_5min_resampled
