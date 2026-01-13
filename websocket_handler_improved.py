import websocket
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import threading
import time
import os
from collections import deque
import logging
from typing import Dict, List, Optional

# ============================================================================
# BINANCE FUTURES WEBSOCKET FEED HANDLER - IMPROVED VERSION
# Handles SOLUSDT 1-min kline stream with 5-min resampling and hourly saves
# ============================================================================
class BinanceWebSocketFeed:
    """
    Production-ready WebSocket feed handler for Binance Futures (Multi-Pair)
    - Subscribes to 1-minute kline stream for MULTIPLE symbols
    - Maintains buffers per symbol
    - Resamples to 5-minute bars on-the-fly
    - Saves parquet rollover every hour
    - Thread-safe with proper locking
    """
    
    def __init__(self, symbols=None, buffer_size=100, 
                 parquet_dir='data/live_buffer', on_5min_bar_callback=None):
        if symbols is None:
            symbols = ['SOLUSDT']
        if isinstance(symbols, str):
            symbols = [symbols]
            
        self.symbols = [s.lower() for s in symbols]
        self.buffer_size = buffer_size
        self.parquet_dir = parquet_dir
        self.on_5min_bar_callback = on_5min_bar_callback
        
        # Thread-safe in-memory buffers (Dict of Deques)
        self.one_min_buffers = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.five_min_buffers = {s: deque(maxlen=buffer_size) for s in self.symbols}
        
        # WebSocket configuration
        # Construct combined stream URL: symbol@kline_1m/symbol2@kline_1m
        streams = "/".join([f"{s}@kline_1m" for s in self.symbols])
        self.ws_url = f"wss://fstream.binance.com/stream?streams={streams}"
        self.ws = None
        self.reconnect_delay = 5
        self.max_reconnect_delay = 300
        self.current_reconnect_delay = self.reconnect_delay
        
        # Timing trackers - Per Symbol
        self.last_5min_bar_timestamps = {s: None for s in self.symbols}
        self.last_processed_timestamps = {s: None for s in self.symbols}
        self.last_parquet_save = datetime.utcnow()
        
        # State lock for thread safety
        self.buffer_lock = threading.Lock()
        self.ws_lock = threading.Lock()
        
        # Connection state
        self.is_connected = False
        self.shutdown_flag = False
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(self.parquet_dir, exist_ok=True)
        
        logging.info(f"Initialized feed handler for {len(self.symbols)} pairs: {', '.join(self.symbols).upper()}")
        
    def _setup_logging(self):
        """Configure production logging"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, f'feed_handler_{self.symbol}.log')
                ),
                logging.StreamHandler()
            ]
        )
    
    def _validate_kline_data(self, kline: Dict) -> bool:
        """Validate incoming kline data"""
        required_fields = ['t', 'o', 'h', 'l', 'c', 'v', 'n', 'q', 'V', 'Q', 'x']
        try:
            for field in required_fields:
                if field not in kline:
                    logging.warning(f"Missing required field: {field}")
                    return False
            
            # Validate numeric fields
            float(kline['o'])
            float(kline['h'])
            float(kline['l'])
            float(kline['c'])
            float(kline['v'])
            
            return True
        except (ValueError, TypeError) as e:
            logging.warning(f"Invalid data format: {e}")
            return False
    
    def on_message(self, ws, message):
        """Process incoming kline messages from combined stream"""
        try:
            payload = json.loads(message)
            
            # Handle subscription confirmation
            if 'result' in payload:
                logging.info(f"Subscription confirmed: {payload}")
                return
            
            # Combined stream format: {"stream": "...", "data": {...}}
            if 'data' not in payload:
                # Could be direct message if single stream, but we use combined stream url now
                data = payload
            else:
                data = payload['data']
            
            # Extract kline data
            if 'k' not in data:
                return
                
            kline = data['k']
            symbol = kline['s'].lower()
            
            # Validate data before processing
            if not self._validate_kline_data(kline):
                logging.warning(f"Data validation failed for {symbol}")
                return
            
            # Only process closed klines
            if kline['x']:
                # Create bar with full Binance payload
                bar = {
                    'symbol': symbol.upper(), # Add symbol to bar data
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'n_trades': int(kline['n']),
                    'quote_volume': float(kline['q']),
                    'taker_buy_base': float(kline['V']),
                    'taker_buy_quote': float(kline['Q']),
                    'is_closed': bool(kline['x'])
                }
                
                # Thread-safe buffer update
                with self.buffer_lock:
                    if symbol in self.one_min_buffers:
                        self.one_min_buffers[symbol].append(bar)
                
                logging.info(
                    f"{symbol.upper()} 1m | {bar['timestamp']} | "
                    f"C:{bar['close']:.4f} V:{bar['volume']:.2f}"
                )
                
                # Trigger resample check using bar timestamp
                self._check_resample(symbol, bar['timestamp'])
                
                # Trigger parquet save check
                self._check_parquet_save()
                
                # Update last processed timestamp
                self.last_processed_timestamps[symbol] = bar['timestamp']
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            logging.error(f"Raw message: {message[:200]}")
        except Exception as e:
            logging.error(f"Message processing error: {e}", exc_info=True)
            logging.error(f"Raw message: {message[:200]}")
            
    def on_error(self, ws, error):
        """Handle WebSocket errors with exponential backoff"""
        logging.error(f"WebSocket error: {error}")
        self.is_connected = False
        
        # Don't attempt reconnection if shutting down
        if self.shutdown_flag:
            return
            
        # Exponential backoff
        logging.info(f"Attempting reconnection in {self.current_reconnect_delay}s...")
        time.sleep(self.current_reconnect_delay)
        
        # Increase delay for next time, up to max
        self.current_reconnect_delay = min(
            self.current_reconnect_delay * 2,
            self.max_reconnect_delay
        )
        
        self._reconnect()
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logging.warning(
            f"WebSocket closed: {close_status_code} - {close_msg}"
        )
        self.is_connected = False
        
        # Attempt reconnection if not shutting down
        if not self.shutdown_flag:
            self._reconnect()
            
    def on_open(self, ws):
        """Handle WebSocket open"""
        logging.info("WebSocket connection established")
        self.is_connected = True
        self.current_reconnect_delay = self.reconnect_delay  # Reset backoff
        
        # Subscribe to kline stream
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [f"{self.symbol}@kline_1m"],
            "id": 1
        }
        ws.send(json.dumps(subscribe_msg))
        
    def _reconnect(self):
        """Safely reconnect WebSocket"""
        with self.ws_lock:
            if self.ws and not self.shutdown_flag:
                logging.info("Reconnecting WebSocket...")
                # Close existing connection if any
                try:
                    self.ws.close()
                except:
                    pass
                # Start new connection
                self.start()
                
                
    def _check_resample(self, symbol, bar_timestamp):
        """
        Resample 1-min to 5-min using actual bar timestamps for specific symbol
        """
        with self.buffer_lock:
            buffer = self.one_min_buffers.get(symbol)
            if not buffer or len(buffer) < 5:
                return
            
            # Get the last 5 bars
            recent_bars = list(buffer)[-5:]
            
            # Check if these 5 bars cover a complete 5-minute period
            first_ts = recent_bars[0]['timestamp']
            last_ts = recent_bars[-1]['timestamp']
            expected_end_ts = first_ts + timedelta(minutes=4, seconds=59)
            
            last_5m_ts = self.last_5min_bar_timestamps.get(symbol)
            
            # If we have a complete 5-min window and it's new
            if (last_ts >= expected_end_ts and 
                (last_5m_ts is None or last_ts > last_5m_ts)):
                
                self._resample_to_5min(symbol, recent_bars)
                self.last_5min_bar_timestamps[symbol] = last_ts
                
    def _resample_to_5min(self, symbol, bars_1m: List[Dict]):
        """Perform 5-minute OHLCV resampling from 1-min bars"""
        try:
            # Create 5-min bar from the 1-min bars
            five_min_bar = {
                'symbol': symbol.upper(),
                'timestamp': bars_1m[0]['timestamp'],
                'open': bars_1m[0]['open'],
                'high': max(b['high'] for b in bars_1m),
                'low': min(b['low'] for b in bars_1m),
                'close': bars_1m[-1]['close'],
                'volume': sum(b['volume'] for b in bars_1m),
                'n_trades': sum(b['n_trades'] for b in bars_1m),
                'quote_volume': sum(b['quote_volume'] for b in bars_1m),
                'taker_buy_base': sum(b['taker_buy_base'] for b in bars_1m),
                'taker_buy_quote': sum(b['taker_buy_quote'] for b in bars_1m)
            }
            
            # Thread-safe buffer update
            with self.buffer_lock:
                if symbol in self.five_min_buffers:
                    self.five_min_buffers[symbol].append(five_min_bar)
            
            logging.info(
                f"{symbol.upper()} 5m | {five_min_bar['timestamp']} | "
                f"C:{five_min_bar['close']:.4f} V:{five_min_bar['volume']:.2f}"
            )
            
            # Call callback if provided
            if self.on_5min_bar_callback:
                try:
                    self.on_5min_bar_callback(five_min_bar)
                except Exception as e:
                    logging.error(f"Callback error: {e}", exc_info=True)
            
        except Exception as e:
            logging.error(f"Resampling error for {symbol}: {e}", exc_info=True)
            
    def _check_parquet_save(self):
        """Save buffers to parquet every hour"""
        current_time = datetime.utcnow()
        
        if current_time >= self.last_parquet_save + timedelta(hours=1):
            self._save_parquet()
            self.last_parquet_save = current_time
            
    def _save_parquet(self):
        """Atomic parquet save with error handling for all symbols"""
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H')
            
            for symbol in self.symbols:
                # Save 1-min buffer
                with self.buffer_lock:
                    if symbol in self.one_min_buffers and self.one_min_buffers[symbol]:
                        df_1m = pd.DataFrame(list(self.one_min_buffers[symbol]))
                        
                        file_path = os.path.join(
                            self.parquet_dir,
                            f"{symbol.lower()}_1m_{timestamp}.parquet"
                        )
                        
                        table = pa.Table.from_pandas(df_1m)
                        pq.write_table(table, file_path)
                        logging.info(f"Saved 1-min parquet: {file_path}")
                        
                # Save 5-min buffer
                with self.buffer_lock:
                    if symbol in self.five_min_buffers and self.five_min_buffers[symbol]:
                        df_5m = pd.DataFrame(list(self.five_min_buffers[symbol]))
                        
                        file_path = os.path.join(
                            self.parquet_dir,
                            f"{symbol.lower()}_5m_{timestamp}.parquet"
                        )
                        
                        table = pa.Table.from_pandas(df_5m)
                        pq.write_table(table, file_path)
                        logging.info(f"Saved 5-min parquet: {file_path}")
                
        except Exception as e:
            logging.error(f"Parquet save failed: {e}", exc_info=True)
            
    def start(self):
        """Start WebSocket connection with daemon thread"""
        with self.ws_lock:
            if self.ws and self.is_connected:
                logging.warning("WebSocket already running")
                return
                
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start in daemon thread
            ws_thread = threading.Thread(target=self._run_forever, daemon=True)
            ws_thread.start()
            
            logging.info(f"Feed handler started for {self.symbol.upper()}")
            
    def _run_forever(self):
        """WebSocket run loop with keepalive"""
        while not self.shutdown_flag:
            try:
                if self.ws:
                    self.ws.run_forever(
                        ping_interval=60,  # Send ping every 60s
                        ping_timeout=10    # Wait 10s for pong
                    )
            except Exception as e:
                logging.error(f"WebSocket run_forever error: {e}", exc_info=True)
                if not self.shutdown_flag:
                    time.sleep(self.reconnect_delay)
                    
    def stop(self):
        """Graceful shutdown"""
        logging.info("Stopping feed handler...")
        self.shutdown_flag = True
        self.is_connected = False
        
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            
        # Final save
        self._save_parquet()
        logging.info("Feed handler stopped")
        
    def get_buffers(self):
        """Thread-safe buffer inspection for all symbols"""
        with self.buffer_lock:
            return {
                'one_min': {s: list(b) for s, b in self.one_min_buffers.items()},
                'five_min': {s: list(b) for s, b in self.five_min_buffers.items()}
            }
            
    def get_status(self):
        """Get current feed status"""
        buffers = self.get_buffers()
        return {
            'connected': self.is_connected,
            'symbols': [s.upper() for s in self.symbols],
            'last_parquet_save': self.last_parquet_save
        }


# ============================================================================
# SAMPLE BUFFER DUMP GENERATOR
# Creates realistic SOLUSDT data for demonstration
# ============================================================================

def generate_sample_buffer():
    """Generate realistic SOLUSDT sample data"""
    base_price = 195.50  # SOL typical price
    current_time = datetime.utcnow().replace(second=0, microsecond=0)
    
    one_min_bars = []
    five_min_bars = []
    
    # Generate 100 1-minute bars (last ~1.6 hours)
    for i in range(100):
        timestamp = current_time - timedelta(minutes=100-i)
        
        # Create realistic price movement with some volatility
        noise = (hash(str(i)) % 1000) / 10000  # Pseudo-random 0-0.1%
        trend = i * 0.0001  # Slight upward trend
        
        # Add some realistic volatility patterns
        volatility = 0.001 if i % 20 != 0 else 0.003  # Higher vol every 20 bars
        
        open_price = base_price + (i * 0.02) + (noise - 0.005)
        close_price = open_price + (trend) + (noise - 0.005) + (volatility * (hash(str(i)) % 100 - 50) / 100)
        high_price = max(open_price, close_price) + abs(noise * 2) + volatility
        low_price = min(open_price, close_price) - abs(noise * 2) - volatility
        volume = 1500 + (hash(str(i)) % 2000)  # 1500-3500 range
        
        one_min_bars.append({
            'timestamp': timestamp,
            'open': round(open_price, 4),
            'high': round(high_price, 4),
            'low': round(low_price, 4),
            'close': round(close_price, 4),
            'volume': round(volume, 2),
            'n_trades': int(volume / 10),  # ~10 SOL per trade
            'quote_volume': round(volume * open_price, 2),
            'taker_buy_base': round(volume * 0.45, 2),  # 45% taker
            'taker_buy_quote': round(volume * 0.45 * open_price, 2),
            'is_closed': True
        })
    
    # Generate 20 5-minute bars (last ~1.6 hours)
    for i in range(20):
        timestamp = current_time - timedelta(minutes=100-i*5)
        
        # Aggregate 5 one-min bars
        start_idx = i * 5
        end_idx = start_idx + 5
        
        if end_idx <= len(one_min_bars):
            bars_subset = one_min_bars[start_idx:end_idx]
            
            five_min_bars.append({
                'timestamp': timestamp,
                'open': bars_subset[0]['open'],
                'high': max(b['high'] for b in bars_subset),
                'low': min(b['low'] for b in bars_subset),
                'close': bars_subset[-1]['close'],
                'volume': round(sum(b['volume'] for b in bars_subset), 2),
                'n_trades': sum(b['n_trades'] for b in bars_subset),
                'quote_volume': round(sum(b['quote_volume'] for b in bars_subset), 2),
                'taker_buy_base': round(sum(b['taker_buy_base'] for b in bars_subset), 2),
                'taker_buy_quote': round(sum(b['taker_buy_quote'] for b in bars_subset), 2)
            })
    
    return one_min_bars, five_min_bars


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if user wants to run live or just see sample
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        print("Generating sample buffer dump...\n")
        one_min, five_min = generate_sample_buffer()
        
        print("=" * 80)
        print("SOL-PERP LIVE FEED BUFFER DUMP")
        print("=" * 80)
        print(f"\n1-Minute Buffer (showing last 10 of {len(one_min)} bars):")
        print("-" * 80)
        print(f"{'Timestamp':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
        print("-" * 80)
        for bar in one_min[-10:]:
            print(f"{bar['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{bar['open']:<8.4f} {bar['high']:<8.4f} {bar['low']:<8.4f} "
                  f"{bar['close']:<8.4f} {bar['volume']:<10.2f}")
        
        print(f"\n5-Minute Resampled Buffer (showing last 5 of {len(five_min)} bars):")
        print("-" * 80)
        print(f"{'Timestamp':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
        print("-" * 80)
        for bar in five_min[-5:]:
            print(f"{bar['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{bar['open']:<8.4f} {bar['high']:<8.4f} {bar['low']:<8.4f} "
                  f"{bar['close']:<8.4f} {bar['volume']:<10.2f}")
        
        print("\n" + "=" * 80)
        print("Parquet files would be saved to: data/live_buffer/")
        print("Hourly rollover format: solusdt_[1m|5m]_YYYYMMDD_HH.parquet")
        print("=" * 80)
        
    else:
        # Live mode
        print("Starting live WebSocket feed handler...")
        print("Press Ctrl+C to stop gracefully\n")
        
        feed = BinanceWebSocketFeed(symbol='SOLUSDT')
        feed.start()
        
        try:
            # Run indefinitely with status updates
            while True:
                time.sleep(30)  # Print status every 30 seconds
                status = feed.get_status()
                print(f"Status: {status['one_min_bars']} 1-min bars, "
                      f"{status['five_min_bars']} 5-min bars, "
                      f"Connected: {status['connected']}")
        except KeyboardInterrupt:
            print("\nShutting down...")
            feed.stop()
