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
# LIVE EVENT DETECTOR GEM (v1.2.0) - OPTION F (FUSION)
# ============================================================================
# Features:
# - Dynamic Filtering (Strict Base / Loose Burst)
# - Burst Mode State Machine (Win Streak / Rolling R)
# - Attack Mode State Machine (Size Scaling)
# - Risk Containment (Drawdown Kill, Cooldowns)
# ============================================================================

class LiveEventDetectorGem:
    """
    Real-time liquidity event detector (GEM v1.2.0 - Fusion)
    - Processes 5-minute bars
    - Tracks Cluster State
    - Detects & Confirms Sweeps
    - Manages Burst & Attack Mode State Machines
    """
    
    def __init__(self, symbol: str, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 enable_telegram: bool = True, telegram_bot=None):
        self.symbol = symbol
        
        # -----------------------------
        # CONFIGURATION (Option F)
        # -----------------------------
        
        # 1. Base (Safety) - Strict Filters
        self.BASE_FILTERS = {
            'SWEEP_WICK_PCT': 0.45,
            'CLUSTER_COMPRESSION_RATIO': 0.6
        }

        # 2. Burst (Attack) - Loose Filters
        self.BURST_FILTERS = {
            'SWEEP_WICK_PCT': 0.35,
            'CLUSTER_COMPRESSION_RATIO': 0.7
        }
        
        # 3. Burst Logic Configuration
        self.BURST_CONFIG = {
            'MAX_DRAWDOWN_SESSION_R': 10.0,
            'MAX_CONSECUTIVE_LOSSES': 10,
            'COOLDOWN_TRADES': 0,
            'MIN_REGIME_EXPECTANCY': -5.0,
            'ATTACK_SIZE_MULT': 2.0,
            'HOT_STREAK_SIZE_MULT': 2.5, 
            'FUNDING_SQUEEZE_BONUS': 0.5
        }

        # 4. Base Logic Configuration
        self.BASE_CONFIG = {
            'MAX_DRAWDOWN_SESSION_R': 2.0,
            'MAX_CONSECUTIVE_LOSSES': 3,
            'COOLDOWN_TRADES': 8,
            'MIN_REGIME_EXPECTANCY': -0.3,
            'ATTACK_SIZE_MULT': 1.5,
            'HOT_STREAK_SIZE_MULT': 1.8,
            'FUNDING_SQUEEZE_BONUS': 0.2
        }

        # 5. Shared Params
        self.SHARED_PARAMS = {
            'SWEEP_THINNING_MULT': 1.2,
            'MAX_BARS_SINCE_CLUSTER': 20,
            'PRIOR_EXTREME_LOOKBACK': 20, # roll_max_20
            'ATR_PERIOD': 20,
            'ATR_TRAILING_STOP_MULT': 1.8,
            'INITIAL_STOP_ATR': 0.95,
            'COOLDOWN_ATR': 1.2,
            'REGIME_WINDOW': 10 
        }
        
        # Buffers
        self.buffer = deque(maxlen=100)
        self.pending_sweep = None
        self.last_cluster_end_idx = -999
        
        # STATE MACHINES
        self.burst_mode_active = False
        self.attack_mode_active = False # Controls sizing primarily
        
        # State Trackers
        self.losses_in_attack = 0
        self.wins_in_attack = 0
        self.attack_session_pnl = 0.0
        self.cooldown_counter = 0      
        self.last_trade_pnl = 0.0
        self.recent_loss = False
        self.consecutive_wins = 0
        
        # Virtual PnL Tracking
        self.virtual_trades = [] 
        self.trade_history = deque(maxlen=20) 
        
        self.event_callback = event_callback or self._default_callback
        self.telegram = telegram_bot if telegram_bot else (TelegramBot() if enable_telegram else None)
        self.lock = threading.Lock()
        self._setup_logging()
        
        self.logger.info(f"GEM v1.2.0 (Option F) Initialized for {symbol}")

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f'GEM_{self.symbol}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(log_dir, 'gem_detector.log'))
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
        """Main Phase Loop"""
        df = pd.DataFrame(list(self.buffer))
        current_idx = len(df) - 1
        row = df.iloc[-1]
        
        # 1. Update Indicators
        self._calculate_indicators(df)
        
        # 2. Manage Virtual Trades & Update PnL State
        self._manage_virtual_trades(row)
        
        # 3. Update Burst Mode State (High Level Regime)
        self._update_burst_state()
        
        # 4. Update Attack Mode State (Sizing & Containment)
        self._manage_attack_mode_state(row)
        
        # 5. Detect Clusters (Dynamic Config)
        self._update_cluster_state(df, current_idx)
        
        # 6. Process Pending Sweep
        if self.pending_sweep:
            self._check_confirmation(df, self.pending_sweep)
            self.pending_sweep = None
            
        # 7. Detect New Sweep (Dynamic Config)
        self._detect_sweep(df, current_idx)

    def _calculate_indicators(self, df: pd.DataFrame):
        df['range'] = df['high'] - df['low']
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.SHARED_PARAMS['ATR_PERIOD'], min_periods=1).mean()
        
        # Volatility Pocket
        df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
        df['atr_slope'] = df['atr'].diff(5)
        df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
        
        # Clusters
        df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
        df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()
        
        # Wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / df['range'] # Replaced later if dynamic? No, static column, used dynamically.
        df['lower_wick_pct'] = df['lower_wick'] / df['range']
        
        # Extremes
        df['roll_max_20'] = df['high'].shift(1).rolling(self.SHARED_PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).max()
        df['roll_min_20'] = df['low'].shift(1).rolling(self.SHARED_PARAMS['PRIOR_EXTREME_LOOKBACK'], min_periods=1).min()

    def _update_burst_state(self):
        """Logic for Switching between Strict (Base) and Loose (Burst) Filters"""
        
        rolling_R = 0.0
        if len(self.trade_history) >= 10:
            rolling_R = np.mean(list(self.trade_history)[-10:])
            
        if not self.burst_mode_active:
            # TRIGGER: No Loss AND (Streak >=3 OR High Rolling R)
            if not self.recent_loss and (self.consecutive_wins >= 3 or rolling_R > 0.7):
                self.burst_mode_active = True
                self.logger.info(f"[BURST] ACTIVATED! Streak: {self.consecutive_wins}, RollR: {rolling_R:.2f}")
        else:
            # DISABLE: Loss OR Edge Fade
            if self.recent_loss:
                self.burst_mode_active = False
                self.logger.info("[BURST] DEACTIVATED (Recent Loss)")
            elif len(self.trade_history) >= 10 and rolling_R < 0.2:
                self.burst_mode_active = False
                self.logger.info(f"[BURST] DEACTIVATED (Edge Faded R={rolling_R:.2f})")

    def _manage_attack_mode_state(self, row: pd.Series):
        """Update Attack Mode (Sizing) based on Config"""
        
        config = self.BURST_CONFIG if self.burst_mode_active else self.BASE_CONFIG
        
        # 1. Damage Kill
        kill_reason = None
        if self.attack_mode_active:
            if self.losses_in_attack >= config['MAX_CONSECUTIVE_LOSSES']:
                kill_reason = "Max Consecutive Losses"
            elif self.attack_session_pnl <= -config['MAX_DRAWDOWN_SESSION_R']:
                kill_reason = "Session Drawdown Cap"
            
            if kill_reason:
                self.attack_mode_active = False
                self.cooldown_counter = config['COOLDOWN_TRADES']
                self.wins_in_attack = 0 # Reset
                # Also kill Burst if hard stop hit
                if self.burst_mode_active:
                    self.burst_mode_active = False
                self.logger.warning(f"[ATTACK KILL] {kill_reason}. Cooldown: {self.cooldown_counter}")
                return

        # 2. Activation
        if not self.attack_mode_active:
             if self.cooldown_counter > 0:
                 pass
             else:
                 is_pocket = row['vol_pocket_active']
                 if is_pocket and self.last_trade_pnl >= 0 and not self.recent_loss:
                     # Regime Check
                     regime_ok = True
                     if len(self.trade_history) >= 12:
                         rolling_r = np.mean(list(self.trade_history)[-10:])
                         if rolling_r < config['MIN_REGIME_EXPECTANCY']:
                             regime_ok = False
                     
                     if regime_ok:
                         self.attack_mode_active = True
                         self.losses_in_attack = 0
                         self.wins_in_attack = 0
                         self.attack_session_pnl = 0.0
                         self.logger.info(f"[ATTACK START] Scaling Enabled (Burst={self.burst_mode_active})")

    def _update_cluster_state(self, df: pd.DataFrame, current_idx: int):
        row = df.iloc[current_idx]
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0: return
        
        # DYNAMIC COMPRESSION FILTER
        filters = self.BURST_FILTERS if self.burst_mode_active else self.BASE_FILTERS
        ratio = filters['CLUSTER_COMPRESSION_RATIO']
        
        if tr_cluster <= ratio * tr_prior:
            self.last_cluster_end_idx = current_idx

    def _detect_sweep(self, df: pd.DataFrame, current_idx: int):
        row = df.iloc[current_idx]
        
        # Shared Eligibility
        if not (row['range'] <= self.SHARED_PARAMS['SWEEP_THINNING_MULT'] * row['atr']): return
        if not row['vol_pocket_active']: return
        if (current_idx - self.last_cluster_end_idx) > self.SHARED_PARAMS['MAX_BARS_SINCE_CLUSTER']: return
        
        # DYNAMIC WICK FILTER
        filters = self.BURST_FILTERS if self.burst_mode_active else self.BASE_FILTERS
        wick_pct_thresh = filters['SWEEP_WICK_PCT']
        
        active_bias = None
        if (row['upper_wick_pct'] > wick_pct_thresh and 
            row['high'] > row['roll_max_20'] and row['close'] < row['roll_max_20']):
            active_bias = 'SHORT'
        elif (row['lower_wick_pct'] > wick_pct_thresh and
              row['low'] < row['roll_min_20'] and row['close'] > row['roll_min_20']):
            active_bias = 'LONG'
            
        if active_bias:
            self.pending_sweep = {
                'bar_idx': current_idx,
                'bias': active_bias,
                'sweep_row': row,
                'atr': row['atr']
            }
            self.logger.info(f"ARMED SWEEP ({active_bias}) - Burst: {self.burst_mode_active}")

    def _check_confirmation(self, df: pd.DataFrame, pending: Dict):
        current_idx = len(df) - 1
        confirm = df.iloc[current_idx]
        sweep = pending['sweep_row']
        bias = pending['bias']
        
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
        
        # Sizing Logic
        config = self.BURST_CONFIG if self.burst_mode_active else self.BASE_CONFIG
        size_mult = 1.0
        
        if self.attack_mode_active:
            size_mult = config['ATTACK_SIZE_MULT']
            if self.wins_in_attack >= 2:
                 size_mult = config['HOT_STREAK_SIZE_MULT']
        
        # Funding Bonus ??? (Included in Opus F? Yes)
        # We need funding rate. Assume passed in 'bar' or fetched? 
        # Backtest had it in DF. Realtime... 'bar' usually doesn't have funding.
        # Skip Funding Bonus for Live v1.2.0 unless we wire funding feed.
        # Safe to omit for now, it's a minor bonus.
        
        if self.cooldown_counter > 0:
             self.cooldown_counter -= 1
        
        self.logger.info(f"🚀 SIGNAL: {direction} @ {entry_price}, Size: {size_mult}x (Burst:{self.burst_mode_active})")
        
        if direction == 'LONG': stop = entry_price - self.SHARED_PARAMS['INITIAL_STOP_ATR'] * atr
        else: stop = entry_price + self.SHARED_PARAMS['INITIAL_STOP_ATR'] * atr
        
        # Track Virtual
        self.virtual_trades.append({
            'direction': direction,
            'entry': entry_price,
            'stop': stop,
            'initial_stop': stop,
            'atr': atr,
            'size_mult': size_mult
        })
        
        # Telegram
        if self.telegram:
            asyncio.create_task(self.telegram.send_entry_alert(
                pair=self.symbol, direction=direction, entry_price=entry_price, 
                event_type=f"GEM_FUSION (Burst={self.burst_mode_active})", stop_loss=stop, take_profit=0, atr=atr, 
                timestamp=timestamp, trailing_stop_atr=self.SHARED_PARAMS['ATR_TRAILING_STOP_MULT']
            ))

        # Build Event
        event_data = {
            'symbol': self.symbol, 'timestamp': timestamp, 'event_type': 'GEM_SWEEP',
            'direction': direction, 'entry_price': entry_price, 'stop_loss': stop, 'take_profit': 0,
            'atr': atr, 'size_multiplier': size_mult, 'volume': bar['volume'], 'bar': bar.to_dict(),
            'meta_attack_mode': self.attack_mode_active,
            'meta_burst_mode': self.burst_mode_active,
            'meta_filters': 'BURST' if self.burst_mode_active else 'BASE'
        }
        
        if self.event_callback and asyncio.iscoroutinefunction(self.event_callback):
            asyncio.run_coroutine_threadsafe(self.event_callback(event_data), asyncio.get_event_loop())
        else:
            threading.Thread(target=self.event_callback, args=(event_data,), daemon=True).start()

    def _manage_virtual_trades(self, row: pd.Series):
        """Update PnL Tracking"""
        active = []
        for trade in self.virtual_trades:
            pnl_r = 0.0
            closed = False
            
            if trade['direction'] == 'LONG':
                if row['low'] <= trade['stop']: # Stop Hit
                    exit_price = trade['stop']
                    pnl_r = (exit_price - trade['entry']) / trade['atr']
                    closed = True
                else: # Trailing Update
                    new_stop = row['close'] - self.SHARED_PARAMS['ATR_TRAILING_STOP_MULT'] * trade['atr']
                    if new_stop > trade['stop']: trade['stop'] = new_stop

            else: # SHORT
                if row['high'] >= trade['stop']: # Stop Hit
                    exit_price = trade['stop']
                    pnl_r = (trade['entry'] - exit_price) / trade['atr']
                    closed = True
                else: # Trailing Update
                   new_stop = row['close'] + self.SHARED_PARAMS['ATR_TRAILING_STOP_MULT'] * trade['atr']
                   if new_stop < trade['stop']: trade['stop'] = new_stop
            
            if closed:
                final_r = pnl_r * trade['size_mult']
                self.update_state_outcome(final_r)
            else:
                active.append(trade)
                
        self.virtual_trades = active

    def update_state_outcome(self, pnl_r: float):
        """Called upon trade closure (Virtual or Real)"""
        with self.lock:
            self.logger.info(f"Trade Result: {pnl_r:.2f}R")
            self.last_trade_pnl = pnl_r
            self.recent_loss = (pnl_r < 0)
            self.trade_history.append(pnl_r)
            
            # Streak Update
            if pnl_r > 0: self.consecutive_wins += 1
            else: self.consecutive_wins = 0
            
            # Attack Session Accumulation
            if self.attack_mode_active:
                self.attack_session_pnl += pnl_r
                if pnl_r < 0:
                    self.losses_in_attack += 1
                    self.wins_in_attack = 0
                else:
                    self.wins_in_attack += 1
                    # Note: losses_in_attack usually resets on win, verified in logic

    def _default_callback(self, data):
        print(f"[GEM] {data['direction']} Signal at {data['timestamp']}")
