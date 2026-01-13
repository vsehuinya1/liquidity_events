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
# LIVE EVENT DETECTOR GEM
# ============================================================================
# Adjusted logic based on 'sweep_cluster_thinning_backtest.py'
# - Primary Signal: Confirmed Sweep
# - Pre-condition: Liquidation Cluster within last 20 bars
# - Validation: Sweep must break 20-bar prior extreme
# - Entry Filter: Confirmation candle must not be too far (Thinning Dist)
# ============================================================================

class LiveEventDetectorGem:
    """
    Real-time liquidity event detector (GEM Strategy)
    - Processes 5-minute bars
    - Tracks Cluster State
    - Detects & Confirms Sweeps
    - 3:1 R:R Logic
    """
    
    def __init__(self, symbol: str, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 enable_telegram: bool = True, telegram_bot=None):
        self.symbol = symbol
        
        # LOCKED PARAMETERS (from backtest)
        self.PARAMS = {
            'SWEEP_RANGE_MULT': 1.8,
            'SWEEP_WICK_PCT': 0.35,
            'SWEEP_VOL_MULT': 1.5,
            'CLUSTER_LEN': 3,
            'THINNING_ATR_DIST': 1.2,
            'MAX_BARS_SINCE_CLUSTER': 20,
            'PRIOR_EXTREME_LOOKBACK': 20,
            'ATR_PERIOD': 20,
            'VOL_MEDIAN_PERIOD': 50
        }
        
        # Buffers
        self.buffer = deque(maxlen=100)  # Need enough for 50 candle lookback + some buffer
        self.pending_sweep = None        # Single active sweep waiting for confirmation
        self.last_cluster_end_idx = -999 # Index of last cluster end
        
        self.event_callback = event_callback or self._default_callback
        
        # Telegram
        self.telegram = telegram_bot if telegram_bot else (TelegramBot() if enable_telegram else None)
        
        # Threading/Logging
        self.lock = threading.Lock()
        self._setup_logging()
        
        self.logger.info("LiveEventDetectorGem initialized with Backtest parameters")

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f'LiveEventDetectorGem_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            # Use a shared log file for all gems, or separate? Shared is better for correlation view.
            # actually better to just log to main file but with prefix
            handler = logging.FileHandler(os.path.join(log_dir, 'event_detector_gem.log'))
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            self.logger.addHandler(handler)

    def on_5min_bar(self, bar: Dict[str, Any]):
        """Ingest new 5-min bar"""
        with self.lock:
            try:
                # Validation
                required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(k in bar for k in required):
                    return

                self.buffer.append(bar)
                
                # Need enough history for calculations (50 for vol median, 20 for extremes)
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
        
        # Update row reference after calculation
        row = df.iloc[-1]
        
        # 2. Detect Clusters (State Update Only)
        self._update_cluster_state(df, current_idx)
        
        # 3. Process Pending Sweep (Confirm/Invalidate)
        if self.pending_sweep:
            self._check_confirmation(df, self.pending_sweep)
            self.pending_sweep = None # Reset after checking one bar
            
        # 4. Detect New Sweep (Arm Only)
        # Only check if we didn't just confirm one (implied by execution order)
        self._detect_sweep(df, current_idx)

    def _calculate_indicators(self, df: pd.DataFrame):
        # Basic
        df['range'] = df['high'] - df['low']
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.PARAMS['ATR_PERIOD'], min_periods=1).mean()
        
        # Wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df['range']
        df['lower_wick_pct'] = df['lower_wick'] / df['range']
        
        # Volume Median
        df['vol_med'] = df['volume'].rolling(self.PARAMS['VOL_MEDIAN_PERIOD'], min_periods=1).median()
        
        # Rolling Extremes (Start of bar)
        # Note: In backtest we used shift(1) because we are looking at *prior* bars.
        # df['roll_max_20'] should comprise bars [i-20...i-1].
        df['roll_max_20'] = df['high'].shift(1).rolling(self.PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).max()
        df['roll_min_20'] = df['low'].shift(1).rolling(self.PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).min()

    def _update_cluster_state(self, df: pd.DataFrame, current_idx: int):
        """Check for cluster completion at current index"""
        cluster_len = self.PARAMS['CLUSTER_LEN']
        if current_idx < cluster_len:
            return

        cluster = df.iloc[current_idx-cluster_len+1 : current_idx+1] # Includes current bar
        # Actually backtest used: cluster = df.iloc[i-CLUSTER_LEN:i] where i was current bar.
        # So it looked at PREVIOUS 3 bars?
        # Backtest loop: row = df.iloc[i]. cluster = df.iloc[i-CLUSTER_LEN:i].
        # So if i=50, cluster is 47,48,49. It checks if *previous* 3 bars formed a cluster.
        # Let's replicate strict timing.
        
        prior_cluster_slice = df.iloc[current_idx-cluster_len : current_idx]
        
        if len(prior_cluster_slice) < cluster_len:
            return

        bull_cluster = (prior_cluster_slice['close'] > prior_cluster_slice['open']).all()
        bear_cluster = (prior_cluster_slice['close'] < prior_cluster_slice['open']).all()

        if bull_cluster or bear_cluster:
            self.last_cluster_end_idx = current_idx - 1 # The cluster ended essentially at previous bar
            # self.logger.info(f"Cluster detected ending at {self.last_cluster_end_idx}")

    def _detect_sweep(self, df: pd.DataFrame, current_idx: int):
        row = df.iloc[current_idx]
        
        # Condition 1: Basic Sweep Metrics
        is_sweep = (
            row['range'] > self.PARAMS['SWEEP_RANGE_MULT'] * row['atr'] and
            row['volume'] > self.PARAMS['SWEEP_VOL_MULT'] * row['vol_med']
        )
        
        if not is_sweep:
            return

        # Condition 2: Cluster Precedence (within 20 bars)
        # Note: current_idx counts bars processed. Logic holds.
        bars_since = current_idx - self.last_cluster_end_idx
        if bars_since > self.PARAMS['MAX_BARS_SINCE_CLUSTER']:
            return # Expired or no cluster

        # Condition 3: Prior Extreme Violation
        prior_max = row['roll_max_20']
        prior_min = row['roll_min_20']
        
        active_bias = None
        
        # Check Short Sweep (Upper Wick)
        if (row['upper_wick_pct'] > self.PARAMS['SWEEP_WICK_PCT'] and 
            row['high'] > prior_max and
            row['close'] < prior_max):
            active_bias = 'SHORT'
            
        # Check Long Sweep (Lower Wick)
        elif (row['lower_wick_pct'] > self.PARAMS['SWEEP_WICK_PCT'] and
              row['low'] < prior_min and 
              row['close'] > prior_min):
            active_bias = 'LONG'
            
        if active_bias:
            self.pending_sweep = {
                'bar_idx': current_idx,
                'bias': active_bias,
                'sweep_row': row,
                'atr': row['atr']
            }
            self.logger.info(f"ARMED SWEEP: {active_bias} at {row['timestamp']} (ATR: {row['atr']:.2f})")

    def _check_confirmation(self, df: pd.DataFrame, pending: Dict):
        """Check next bar for failed continuation + distance filter"""
        current_idx = len(df) - 1
        
        # We expect confirmation on the VERY NEXT bar
        if current_idx != pending['bar_idx'] + 1:
            return # Should not happen with this periodic logic, but safe guard

        confirm = df.iloc[current_idx]
        sweep = pending['sweep_row']
        bias = pending['bias']
        
        # 1. Failed Continuation
        failed = False
        if bias == 'SHORT':
            failed = confirm['close'] < sweep['close']
        else:
            failed = confirm['close'] > sweep['close']
            
        if not failed:
            self.logger.info(f"Sweep {pending['bar_idx']} failed confirmation (no reversal)")
            return

        # 2. Thinning Distance Filter
        # "dist = abs(confirm['close'] - row['close'])"
        dist = abs(confirm['close'] - sweep['close'])
        if dist > self.PARAMS['THINNING_ATR_DIST'] * sweep['atr']:
             self.logger.info(f"Sweep {pending['bar_idx']} filtered by Thinning Distance ({dist:.2f} > limit)")
             return

        # 3. SUCCESS - EMIT SIGNAL
        self._emit_signal(confirm, bias, sweep['atr'])

    def _emit_signal(self, bar: pd.Series, direction: str, atr: float):
        timestamp = bar['timestamp']
        entry_price = bar['close']
        
        self.logger.info(f"🚀 SIGNAL CONFIRMED: {direction} at {entry_price} (Time: {timestamp})")
        
        # R:R 3:1 Logic
        if direction == 'LONG':
            stop = entry_price - atr
            target = entry_price + 3 * atr
        else:
            stop = entry_price + atr
            target = entry_price - 3 * atr
            
        # Telegram Alert
        if self.telegram:
            try:
                # We spin off a task for async items
                 asyncio.create_task(
                    self.telegram.send_entry_alert(
                        pair=self.symbol,
                        direction=direction,
                        entry_price=entry_price,
                        event_type="GEM_SWEEP", # Distinguish our event
                        stop_loss=stop,
                        take_profit=target,
                        atr=atr,
                        timestamp=timestamp
                    )
                )
            except Exception as e:
                self.logger.error(f"Telegram error: {e}")

        # Callback
        event_data = {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'event_type': 'GEM_SWEEP',
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop,
            'take_profit': target,
            'atr': atr,
            'volume': bar['volume'],
            'bar': bar.to_dict()
        }
        
        if self.event_callback and asyncio.iscoroutinefunction(self.event_callback):
            asyncio.run_coroutine_threadsafe(self.event_callback(event_data), asyncio.get_event_loop())
        else:
            threading.Thread(target=self.event_callback, args=(event_data,), daemon=True).start()

    def _default_callback(self, data):
        print(f"[GEM] {data['direction']} Signal at {data['timestamp']}")

# Helper for integration
def integrate_with_feed_handler_gem(feed_handler, detector):
    def on_5min_resampled(bar):
        detector.on_5min_bar(bar)
    return on_5min_resampled
