# live_event_detector.py
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
from collections import deque
import json
import time

# ============================================================================
# LIVE EVENT DETECTOR
# ============================================================================
# Copy-paste logic from master_thinning_strategy.py
# Runs on 5-min buffer, triggers event flags
# Logs: timestamp, event_type, ATR, volume
# Sends Telegram alerts for paper trading
# ============================================================================

class LiveEventDetector:
    """
    Real-time liquidity event detector for SOL-PERP
    - Processes 5-minute bars from feed handler
    - Calculates rolling metrics (ATR, wicks, volume)
    - Detects sweeps, clusters, and thinning events
    - Thread-safe and production-ready
    - Sends Telegram alerts for paper trading
    """
    
    def __init__(self, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                 enable_telegram: bool = True, telegram_bot=None):
        # Parameters directly from locked script
        self.SWEEP_PARAMS = {
            'range_mult': 1.8,
            'wick_pct': 0.35,
            'volume_mult': 1.5,
            'lookback_range': 5
        }
        self.CLUSTER_PARAMS = {
            'consecutive_candles': 3,
            'cumulative_atr_mult': 2.5,
            'exhaustion_volume_drop': 0.6,
            'failure_to_extend': True
        }
        self.THINNING_PARAMS = {
            'range_expansion_mult': 1.6,
            'volume_divergence': 0.7,
            'subsequent_wick_threshold': 0.30,
            'rapid_alternation': 3
        }
        
        # Thread-safe buffers
        self.buffer = deque(maxlen=100)  # 5-min bars with metrics
        self.pending_thinning_events = []  # Pending thinning confirmations
        self.pending_cluster_events = []   # Pending cluster confirmations
        self.pending_sweep_events = []     # Pending sweep confirmations
        self.event_callback = event_callback or self._default_callback
        
        # Decay radar configuration
        self.enable_decay_filter = True
        self.decay_threshold = 0.75  # From params file
        
        # Decay metric buffers
        self._init_decay_metrics()
        
        # Telegram integration for paper trading alerts
        self.telegram = telegram_bot if telegram_bot else (TelegramBot() if enable_telegram else None)
        
        # Synchronization
        self.lock = threading.Lock()
        
        # Logging
        self._setup_logging()
        
        self.logger.info("LiveEventDetector with decay radar initialized")

    def _init_decay_metrics(self):
        """Initialize decay tracking buffers"""
        self.sol_decay_buffer = deque(maxlen=48)  # 4 hours of 5-min bars
        self.last_decay_score = 0.0

    def _update_decay_metrics(self, bar: Dict[str, Any]):
        """Calculate decay sub-scores after each bar"""
        if len(self.buffer) < 48:
            return
        
        df = pd.DataFrame(list(self.buffer))
        latest = df.iloc[-1]
        
        # Sub-signal 1: Micro-volatility surge
        # FIX: Use buffer-appropriate window (max 96 bars, but buffer is 100)
        vol_baseline = df['range'].rolling(min(len(df), 96), min_periods=20).median().iloc[-1]
        vol_short = df['range'].tail(12).std()
        vol_score = min(vol_short / (vol_baseline + 1e-6), 1.0)
        
        # Sub-signal 2: Wick alternation
        wick_alt = ((df['upper_wick_pct'] > 0.25) & (df['lower_wick_pct'].shift() > 0.25)).tail(6).sum()
        alt_score = wick_alt / 6
        
        # Sub-signal 3: Funding shock proxy (FIXED: graduated instead of binary)
        avg_vol_48 = df['volume'].rolling(48, min_periods=12).quantile(0.9).iloc[-1]
        vol_spike = latest['volume'] / (avg_vol_48 + 1e-6)
        body_ratio = abs(latest['close'] - latest['open']) / (latest['range'] + 1e-6)
        
        # Higher score when: high volume + small body (absorption/distribution)
        if vol_spike > 1.0 and body_ratio < 0.3:
            fund_score = min((vol_spike - 1.0) * 0.5, 1.0)  # Scale up to 1.0
        else:
            fund_score = 0.0
        
        # Sub-signal 4: Range compression (FIXED: correct logic)
        range_atr = latest['range'] / (latest['atr_20'] + 1e-6)
        hist_ratio = df['range'].tail(48).median() / (df['atr_20'].tail(48).median() + 1e-6)
        
        # When range < historical (compressed), ratio < 1, score approaches 1
        compress_score = max(0, 1 - (range_atr / (hist_ratio + 1e-6)))
        compress_score = min(compress_score, 1.0)  # Cap at 1.0
        
        # Composite score
        self.last_decay_score = (
            0.35 * vol_score +
            0.25 * alt_score +
            0.20 * fund_score +
            0.20 * compress_score
        )
        
        if self.last_decay_score > 0.65:
            self.logger.warning(
                f"SOL decay alert: {self.last_decay_score:.2f} "
                f"(vol={vol_score:.2f}, alt={alt_score:.2f}, fund={fund_score:.2f}, compress={compress_score:.2f})"
            )

    def _should_skip_trade(self) -> bool:
        """Gate function: skip if decay score exceeds threshold"""
        if not self.enable_decay_filter:
            return False
        
        if self.last_decay_score >= self.decay_threshold:
            self.logger.info(f"Decay filter ACTIVE (score={self.last_decay_score:.2f}) - skipping event")
            return True
        
        return False
        
    def _setup_logging(self):
        """Configure production logging"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(
                os.path.join(log_dir, 'event_detector.log')
            )
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def on_5min_bar(self, bar: Dict[str, Any]):
        """
        Process new 5-minute bar from feed handler
        Called externally when 5-min bar completes
        """
        with self.lock:
            try:
                # Validate bar
                required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(f in bar for f in required_fields):
                    self.logger.error(f"Invalid bar format: {list(bar.keys())}")
                    return
                
                # Append to buffer
                self.buffer.append(bar)
                
                # Calculate metrics if we have enough data
                if len(self.buffer) >= 20:
                    self._update_metrics()
                    self._run_detections()
                    self._process_pending_events()
                    
            except Exception as e:
                self.logger.error(f"Error processing bar: {e}", exc_info=True)
    
    def _update_metrics(self):
        """Calculate ATR, wicks, volume metrics + decay signals"""
        df = pd.DataFrame(list(self.buffer))
        
        # Existing calculations
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['atr_20'] = self._calculate_atr(df)
        df['volume_median_50'] = df['volume'].rolling(50, min_periods=1).median()
        df['upper_wick_pct'] = df['upper_wick'] / df['range']
        df['lower_wick_pct'] = df['lower_wick'] / df['range']
        df['upper_wick_pct'] = df['upper_wick_pct'].replace([np.inf, -np.inf], 0)
        df['lower_wick_pct'] = df['lower_wick_pct'].replace([np.inf, -np.inf], 0)
        
        # Update buffer FIRST
        for i, (_, row) in enumerate(df.iterrows()):
            self.buffer[i].update(row.to_dict())
        
        # THEN update decay metrics with the updated buffer (FIX: timing)
        if len(self.buffer) >= 48:
            self._update_decay_metrics(self.buffer[-1])
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """ATR calculation from master_thinning_strategy.py"""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    def _run_detections(self):
        """Run all three detection algorithms"""
        # Sweeps and clusters can be detected immediately
        self._detect_liquidity_sweeps()
        self._detect_liquidation_clusters()
        
        # Thinning initiates pending event
        self._detect_liquidity_thinning()
    
    def _detect_liquidity_sweeps(self):
        """
        COPY-PASTE LOGIC from master_thinning_strategy.py
        Detects liquidity sweeps with range/volume/wick criteria
        """
        df = pd.DataFrame(list(self.buffer))
        i = len(df) - 1
        
        if i < self.SWEEP_PARAMS['lookback_range']:
            return
        
        current = df.iloc[i]
        prior_start = max(0, i - self.SWEEP_PARAMS['lookback_range'])
        prior_bars = df.iloc[prior_start:i]
        
        # Exact conditions from backtest
        range_condition = current['range'] >= (self.SWEEP_PARAMS['range_mult'] * current['atr_20'])
        volume_condition = current['volume'] >= (self.SWEEP_PARAMS['volume_mult'] * current['volume_median_50'])
        upper_wick_condition = current['upper_wick_pct'] >= self.SWEEP_PARAMS['wick_pct']
        lower_wick_condition = current['lower_wick_pct'] >= self.SWEEP_PARAMS['wick_pct']
        wick_condition = upper_wick_condition or lower_wick_condition
        
        if range_condition and volume_condition and wick_condition:
            direction = 'UPPER' if upper_wick_condition else 'LOWER'
            # ARM the event - sweeps are information, not immediate entries
            self.pending_sweep_events.append({
                'bar_idx': i,
                'direction': direction,
                'is_upper': upper_wick_condition,
                'atr': current['atr_20'],
                'volume': current['volume'],
                'timestamp': current['timestamp']
            })
    
    def _detect_liquidation_clusters(self):
        """
        COPY-PASTE LOGIC from master_thinning_strategy.py
        Detects liquidation clusters with exhaustion confirmation
        """
        df = pd.DataFrame(list(self.buffer))
        i = len(df) - 1
        
        # Need current bar + previous cluster
        if i < self.CLUSTER_PARAMS['consecutive_candles'] + 1:
            return
        
        # Cluster ends at i-1, exhaustion at i
        cluster_end_idx = i - 1
        cluster_start_idx = max(0, cluster_end_idx - self.CLUSTER_PARAMS['consecutive_candles'] + 1)
        cluster_bars = df.iloc[cluster_start_idx:cluster_end_idx + 1]
        
        if len(cluster_bars) < self.CLUSTER_PARAMS['consecutive_candles']:
            return
        
        # Cluster direction
        bullish_cluster = all(cluster_bars['close'] > cluster_bars['open'])
        bearish_cluster = all(cluster_bars['close'] < cluster_bars['open'])
        
        if not (bullish_cluster or bearish_cluster):
            return
        
        # Exhaustion conditions on current bar
        current_bar = df.iloc[i]
        cluster_avg_volume = cluster_bars['volume'].mean()
        cluster_avg_range = cluster_bars['range'].mean()
        
        volume_drop = current_bar['volume'] < (self.CLUSTER_PARAMS['exhaustion_volume_drop'] * cluster_avg_volume)
        failure_to_extend = current_bar['range'] < cluster_avg_range
        
        if volume_drop and failure_to_extend:
            # ARM the event - don't emit yet, wait for confirmation
            self.pending_cluster_events.append({
                'bar_idx': i,
                'direction': 'BULLISH' if bullish_cluster else 'BEARISH',
                'is_bullish': bullish_cluster,
                'atr': current_bar['atr_20'],
                'volume': current_bar['volume'],
                'timestamp': current_bar['timestamp']
            })
    
    def _detect_liquidity_thinning(self):
        """
        COPY-PASTE LOGIC from master_thinning_strategy.py
        Initiates pending events for range expansion + volume divergence
        """
        df = pd.DataFrame(list(self.buffer))
        i = len(df) - 1
        
        if i < 10:  # Need 10 prior bars
            return
        
        current = df.iloc[i]
        prior_bars = df.iloc[i-10:i]
        
        # Initial thinning conditions
        avg_prior_range = prior_bars['range'].mean()
        avg_prior_volume = prior_bars['volume'].mean()
        
        range_expansion = current['range'] >= (self.THINNING_PARAMS['range_expansion_mult'] * avg_prior_range)
        volume_divergence = current['volume'] < (self.THINNING_PARAMS['volume_divergence'] * avg_prior_volume * self.THINNING_PARAMS['range_expansion_mult'])
        
        if range_expansion and volume_divergence:
            # Store pending event for confirmation
            self.pending_thinning_events.append({
                'bar_idx': i,
                'timestamp': current['timestamp'],
                'atr': current['atr_20'],
                'volume': current['volume']
            })
    
    def _process_pending_events(self):
        """Check pending thinning events against subsequent bars"""
        df = pd.DataFrame(list(self.buffer))
        current_idx = len(df) - 1
        
        still_pending = []
        
        for event in self.pending_thinning_events:
            bar_idx = event['bar_idx']
            
            # Check if within 3-bar confirmation window
            if current_idx <= bar_idx + 3:
                # Look for large wicks in bars after the event
                subsequent_bars = df.iloc[bar_idx+1:current_idx+1]
                
                if not subsequent_bars.empty:
                    large_wicks = (subsequent_bars['upper_wick_pct'] >= self.THINNING_PARAMS['subsequent_wick_threshold']).any() or \
                                 (subsequent_bars['lower_wick_pct'] >= self.THINNING_PARAMS['subsequent_wick_threshold']).any()
                    
                    if large_wicks:
                        # Confirmed - emit event for original bar
                        self._emit_event('liquidity_thinning', df.iloc[bar_idx])
                        continue  # Don't keep this event
                
                still_pending.append(event)
            # Events older than 3 bars are discarded
        
        self.pending_thinning_events = still_pending
        
        # NEW: Process pending cluster events
        still_pending_clusters = []
        for event in self.pending_cluster_events:
            bar_idx = event['bar_idx']
            
            if current_idx == bar_idx + 1:  # Next bar after exhaustion
                confirm_bar = df.iloc[current_idx]
                
                # Confirm reversal: opposite direction candle
                if event['is_bullish']:  # Was bullish cluster
                    # Confirm with bearish candle (close < open)
                    if confirm_bar['close'] < confirm_bar['open']:
                        self._emit_event('liquidation_cluster', confirm_bar, direction=event['direction'])
                        continue  # Confirmed, don't keep
                else:  # Was bearish cluster
                    # Confirm with bullish candle (close > open)
                    if confirm_bar['close'] > confirm_bar['open']:
                        self._emit_event('liquidation_cluster', confirm_bar, direction=event['direction'])
                        continue  # Confirmed, don't keep
                
                still_pending_clusters.append(event)
            elif current_idx > bar_idx + 2:  # More than 2 bars old, discard
                pass
            else:
                still_pending_clusters.append(event)
        
        self.pending_cluster_events = still_pending_clusters
        
        # NEW: Process pending sweep events
        still_pending_sweeps = []
        for event in self.pending_sweep_events:
            bar_idx = event['bar_idx']
            
            if current_idx == bar_idx + 1:  # Next bar after sweep
                confirm_bar = df.iloc[current_idx]
                
                # Confirm reversal from the sweep direction
                if event['is_upper']:  # Was upper wick sweep
                    # Confirm with bearish follow-through (close < open OR close below sweep bar close)
                    sweep_bar = df.iloc[bar_idx]
                    if confirm_bar['close'] < confirm_bar['open'] or confirm_bar['close'] < sweep_bar['close']:
                        self._emit_event('liquidity_sweep', confirm_bar, direction=event['direction'])
                        continue  # Confirmed
                else:  # Was lower wick sweep
                    # Confirm with bullish follow-through (close > open OR close above sweep bar close)
                    sweep_bar = df.iloc[bar_idx]
                    if confirm_bar['close'] > confirm_bar['open'] or confirm_bar['close'] > sweep_bar['close']:
                        self._emit_event('liquidity_sweep', confirm_bar, direction=event['direction'])
                        continue  # Confirmed
                
                still_pending_sweeps.append(event)
            elif current_idx > bar_idx + 2:  # More than 2 bars old, discard
                pass
            else:
                still_pending_sweeps.append(event)
        
        self.pending_sweep_events = still_pending_sweeps
    
    def _emit_event(self, event_type: str, bar_data: pd.Series, direction: Optional[str] = None):
        """Log and callback for detected events"""
        # FIX: Check decay filter BEFORE emitting event
        if self._should_skip_trade():
            return  # Silently skip when decay score is high
        
        timestamp = bar_data['timestamp']
        atr_val = bar_data['atr_20']
        volume_val = bar_data['volume']
        
        # REQUIRED LOG FORMAT: timestamp, event_type, ATR, volume
        log_msg = f"{timestamp} | {event_type:<18} | ATR:{atr_val:.4f} | Volume:{volume_val:.2f}"
        if direction:
            log_msg += f" | Dir:{direction}"
        
        self.logger.info(log_msg)
        
        # Send Telegram alert for paper trading
        if self.telegram:
            # Determine trade direction based on event type and direction
            trade_direction = "LONG"
            if event_type == 'liquidity_sweep':
                trade_direction = "SHORT" if direction == "UPPER" else "LONG"
            elif event_type == 'liquidation_cluster':
                trade_direction = "SHORT" if direction == "BULLISH" else "LONG"
            elif event_type == 'liquidity_thinning':
                # Trade WITH the wick rejection direction
                # Upper wick rejection (sellers won) → SHORT
                # Lower wick rejection (buyers won) → LONG
                if len(self.buffer) >= 3:
                    recent_bars = list(self.buffer)[-3:]  # Look at confirming bars
                    for bar in recent_bars[1:]:  # Skip original expansion bar
                        if bar.get('upper_wick_pct', 0) >= 0.30:
                            trade_direction = "SHORT"  # Upper wick rejection
                            break
                        elif bar.get('lower_wick_pct', 0) >= 0.30:
                            trade_direction = "LONG"   # Lower wick rejection
                            break
            
            # Calculate SL/TP based on ATR (1:3 risk/reward)
            entry_price = bar_data['close']
            sl_distance = atr_val * 1.5  # 1.5x ATR stop
            
            if trade_direction == "LONG":
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + (sl_distance * 3)
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - (sl_distance * 3)
            
            # Send async Telegram alert
            try:
                asyncio.create_task(
                    self.telegram.send_entry_alert(
                        pair="SOLUSDT",
                        direction=trade_direction,
                        entry_price=entry_price,
                        event_type=event_type,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        atr=atr_val,
                        timestamp=timestamp
                    )
                )
            except Exception as e:
                self.logger.error(f"Failed to send Telegram alert: {e}")
        
        # Callback for downstream systems
        event_data = {
            'timestamp': timestamp,
            'event_type': event_type,
            'atr': atr_val,
            'volume': volume_val,
            'direction': direction,
            'bar': bar_data.to_dict(),
            'decay_score': self.last_decay_score
        }
        
        # Call async callback properly from sync context
        if self.event_callback and asyncio.iscoroutinefunction(self.event_callback):
            # Schedule async callback on the event loop
            asyncio.run_coroutine_threadsafe(self.event_callback(event_data), asyncio.get_event_loop())
        else:
            # Fallback for sync callbacks
            threading.Thread(target=self.event_callback, args=(event_data,), daemon=True).start()
    
    def _default_callback(self, event_data: Dict[str, Any]):
        """Default handler - can be overridden"""
        print(f"[EVENT] {event_data['event_type']} at {event_data['timestamp']} (ATR: {event_data['atr']:.4f})")


# ============================================================================
# INTEGRATION WITH FEED HANDLER
# ============================================================================

def integrate_with_feed_handler(feed_handler, detector):
    """
    Connect feed handler to event detector
    feed_handler: BinanceWebSocketFeed instance
    detector: LiveEventDetector instance
    """
    
    def on_5min_resampled(bar):
        """Bridge function called by feed handler when 5-min bar completes"""
        detector.on_5min_bar(bar)
    
    # This would be called in feed_handler._resample_to_5min()
    return on_5min_resampled


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create detector with custom callback
    def my_event_handler(event):
        print(f"\n🚨 EVENT DETECTED 🚨")
        print(f"Type: {event['event_type']}")
        print(f"Time: {event['timestamp']}")
        print(f"ATR: {event['atr']:.4f}")
        print(f"Volume: {event['volume']:.2f}")
        if event['direction']:
            print(f"Direction: {event['direction']}")
        print("=" * 50)
    
    detector = LiveEventDetector(event_callback=my_event_handler)
    
    # Simulate 5-min bars from feed handler
    print("Simulating live 5-min bars...\n")
    
    # Example sequence that should trigger events
    base_price = 195.50
    
    for i in range(15):
        # Create bar with specific patterns
        bar = {
            'timestamp': datetime.utcnow(),
            'open': base_price + i * 0.1,
            'high': base_price + i * 0.1 + 0.5,
            'low': base_price + i * 0.1 - 0.5,
            'close': base_price + i * 0.1 + 0.1,
            'volume': 2000 + i * 100
        }
        
        # Inject sweep pattern at bar 5
        if i == 5:
            bar['high'] = base_price + 2.0  # Large range
            bar['volume'] = 5000  # High volume
            bar['close'] = base_price - 0.5  # Large upper wick
        
        # Inject thinning pattern at bar 8
        if i == 8:
            bar['range'] = 1.5  # Range expansion
            bar['volume'] = 1000  # Volume divergence
        
        detector.on_5min_bar(bar)
        print(f"Processed bar {i+1}")
        time.sleep(0.1)
