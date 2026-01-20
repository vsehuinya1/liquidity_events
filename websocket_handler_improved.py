# websocket_handler_improved.py
"""
Binance Futures WebSocket Feed Handler - Refactored
Plumbing only: connect, normalize bars, emit bars
"""

import websocket
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import threading
import time
import os
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

DEFAULT_BUFFER_SIZE = 100
DEFAULT_PARQUET_DIR = 'data/live_buffer'
RECONNECT_DELAY_INITIAL_SEC = 5
RECONNECT_DELAY_MAX_SEC = 300
PING_INTERVAL_SEC = 60
PING_TIMEOUT_SEC = 10
PARQUET_SAVE_INTERVAL_HOURS = 1
REQUIRED_KLINE_FIELDS = ('t', 'o', 'h', 'l', 'c', 'v', 'n', 'q', 'V', 'Q', 'x')


@dataclass
class WebSocketState:
    connected: bool = False
    running: bool = False
    reconnect_attempts: int = 0
    current_reconnect_delay: float = RECONNECT_DELAY_INITIAL_SEC
    last_parquet_save: datetime = field(default_factory=datetime.utcnow)
    last_bar_timestamps: Dict[str, Optional[int]] = field(default_factory=dict)


def parse_raw_message(raw_message: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw_message)
    except json.JSONDecodeError:
        return None


def extract_kline_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if 'result' in payload:
        return None
    if 'data' in payload:
        data = payload['data']
    else:
        data = payload
    if 'k' not in data:
        return None
    return data['k']


def extract_symbol_from_kline(kline: Dict[str, Any]) -> str:
    return kline.get('s', '').lower()


def validate_kline_fields(kline: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    for field_name in REQUIRED_KLINE_FIELDS:
        if field_name not in kline:
            return False, f"Missing required field: {field_name}"
    return True, None


def validate_kline_numerics(kline: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        float(kline['o'])
        float(kline['h'])
        float(kline['l'])
        float(kline['c'])
        float(kline['v'])
        return True, None
    except (ValueError, TypeError) as e:
        return False, f"Invalid numeric format: {e}"


def validate_bar_integrity(open_price: float, high_price: float, low_price: float, close_price: float, volume: float) -> Tuple[bool, Optional[str]]:
    if high_price < low_price:
        return False, f"Invalid OHLC: high ({high_price}) < low ({low_price})"
    bar_range = high_price - low_price
    if volume == 0 and bar_range > 0:
        return False, f"Zero volume with nonzero range ({bar_range})"
    return True, None


def is_new_bar(bar_timestamp_ms: int, last_bar_timestamp_ms: Optional[int]) -> bool:
    if last_bar_timestamp_ms is None:
        return True
    return bar_timestamp_ms > last_bar_timestamp_ms


def is_bar_closed(kline: Dict[str, Any]) -> bool:
    return bool(kline.get('x', False))


def extract_ohlcv_bar(kline: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    return {
        'symbol': symbol.upper(),
        'timestamp': pd.to_datetime(kline['t'], unit='ms'),
        'timestamp_ms': int(kline['t']),
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


def can_resample_5min(buffer_length: int, first_bar_ts: datetime, last_bar_ts: datetime, last_5min_ts: Optional[datetime]) -> bool:
    if buffer_length < 5:
        return False
    expected_end = first_bar_ts + timedelta(minutes=4, seconds=59)
    if last_bar_ts < expected_end:
        return False
    if last_5min_ts is not None and last_bar_ts <= last_5min_ts:
        return False
    return True


def resample_bars_to_5min(bars_1m: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
    return {
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


def build_stream_url(symbols: List[str]) -> str:
    streams = "/".join([f"{s}@kline_1m" for s in symbols])
    return f"wss://fstream.binance.com/stream?streams={streams}"


def setup_feed_logging() -> logging.Logger:
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'feed_handler_multipair.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('WebSocketFeed')


class BinanceWebSocketFeed:
    def __init__(self, symbols: Optional[List[str]] = None, buffer_size: int = DEFAULT_BUFFER_SIZE, parquet_dir: str = DEFAULT_PARQUET_DIR, on_5min_bar_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        if symbols is None:
            symbols = ['SOLUSDT']
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = [s.lower() for s in symbols]
        self.buffer_size = buffer_size
        self.parquet_dir = parquet_dir
        self.on_5min_bar_callback = on_5min_bar_callback
        self.ws_url = build_stream_url(self.symbols)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.state = WebSocketState(last_bar_timestamps={s: None for s in self.symbols})
        self.one_min_buffers: Dict[str, deque] = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.five_min_buffers: Dict[str, deque] = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.last_5min_bar_timestamps: Dict[str, Optional[datetime]] = {s: None for s in self.symbols}
        self.buffer_lock = threading.Lock()
        self.ws_lock = threading.Lock()
        self.logger = setup_feed_logging()
        os.makedirs(self.parquet_dir, exist_ok=True)
        self.logger.info(f"Initialized feed handler for {len(self.symbols)} pairs: {', '.join(self.symbols).upper()}")

    @property
    def is_connected(self) -> bool:
        return self.state.connected
    
    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self.state.connected = value
    
    @property
    def shutdown_flag(self) -> bool:
        return not self.state.running
    
    @shutdown_flag.setter
    def shutdown_flag(self, value: bool) -> None:
        self.state.running = not value
    
    @property
    def last_parquet_save(self) -> datetime:
        return self.state.last_parquet_save
    
    @last_parquet_save.setter
    def last_parquet_save(self, value: datetime) -> None:
        self.state.last_parquet_save = value
    
    @property
    def reconnect_delay(self) -> float:
        return RECONNECT_DELAY_INITIAL_SEC
    
    @property
    def max_reconnect_delay(self) -> float:
        return RECONNECT_DELAY_MAX_SEC
    
    @property
    def current_reconnect_delay(self) -> float:
        return self.state.current_reconnect_delay
    
    @current_reconnect_delay.setter
    def current_reconnect_delay(self, value: float) -> None:
        self.state.current_reconnect_delay = value

    def start(self) -> None:
        with self.ws_lock:
            if self.ws and self.state.connected:
                self.logger.warning("WebSocket already running")
                return
            self.state.running = True
            self.ws = websocket.WebSocketApp(self.ws_url, on_open=self._on_open, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
            ws_thread = threading.Thread(target=self._run_forever, daemon=True)
            ws_thread.start()
            self.logger.info(f"Feed handler started for {len(self.symbols)} pairs")

    def stop(self) -> None:
        self.logger.info("Stopping feed handler...")
        self.state.running = False
        self.state.connected = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self._save_parquet()
        self.logger.info("Feed handler stopped")

    def reconnect(self) -> None:
        with self.ws_lock:
            if not self.state.running:
                return
            self.logger.info("Reconnecting WebSocket...")
            self.state.reconnect_attempts += 1
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
            self.start()

    def _on_open(self, ws) -> None:
        self.logger.info("WebSocket connection established")
        self.state.connected = True
        self.state.current_reconnect_delay = RECONNECT_DELAY_INITIAL_SEC
        self.state.reconnect_attempts = 0

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.state.connected = False
        if self.state.running:
            self.reconnect()

    def _on_error(self, ws, error) -> None:
        self.logger.error(f"WebSocket error: {error}")
        self.state.connected = False
        if not self.state.running:
            return
        self.logger.info(f"Attempting reconnection in {self.state.current_reconnect_delay}s...")
        time.sleep(self.state.current_reconnect_delay)
        self.state.current_reconnect_delay = min(self.state.current_reconnect_delay * 2, RECONNECT_DELAY_MAX_SEC)
        self.reconnect()

    def _on_message(self, ws, message: str) -> None:
        payload = parse_raw_message(message)
        if payload is None:
            self.logger.error(f"JSON decode error. Raw: {message[:200]}")
            return
        kline = extract_kline_from_payload(payload)
        if kline is None:
            if 'result' in payload:
                self.logger.info(f"Subscription confirmed: {payload}")
            return
        symbol = extract_symbol_from_kline(kline)
        if symbol not in self.symbols:
            return
        is_valid, error = validate_kline_fields(kline)
        if not is_valid:
            self.logger.warning(f"Field validation failed for {symbol}: {error}")
            return
        is_valid, error = validate_kline_numerics(kline)
        if not is_valid:
            self.logger.warning(f"Numeric validation failed for {symbol}: {error}")
            return
        if not is_bar_closed(kline):
            return
        bar_timestamp_ms = int(kline['t'])
        last_timestamp_ms = self.state.last_bar_timestamps.get(symbol)
        if not is_new_bar(bar_timestamp_ms, last_timestamp_ms):
            self.logger.debug(f"Dropped duplicate/old bar for {symbol}: ts={bar_timestamp_ms}, last={last_timestamp_ms}")
            return
        bar = extract_ohlcv_bar(kline, symbol)
        is_valid, error = validate_bar_integrity(bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
        if not is_valid:
            self.logger.warning(f"Bar integrity failed for {symbol}: {error}")
            return
        self.state.last_bar_timestamps[symbol] = bar_timestamp_ms
        with self.buffer_lock:
            self.one_min_buffers[symbol].append(bar)
        self.logger.info(f"{symbol.upper()} 1m | {bar['timestamp']} | C:{bar['close']:.4f} V:{bar['volume']:.2f}")
        self._try_resample(symbol, bar['timestamp'])
        self._check_parquet_save()

    def _run_forever(self) -> None:
        while self.state.running:
            try:
                if self.ws:
                    self.ws.run_forever(ping_interval=PING_INTERVAL_SEC, ping_timeout=PING_TIMEOUT_SEC)
            except Exception as e:
                self.logger.error(f"WebSocket run_forever error: {e}", exc_info=True)
                if self.state.running:
                    time.sleep(RECONNECT_DELAY_INITIAL_SEC)

    def _try_resample(self, symbol: str, bar_timestamp: datetime) -> None:
        with self.buffer_lock:
            buffer = self.one_min_buffers.get(symbol)
            if not buffer or len(buffer) < 5:
                return
            recent_bars = list(buffer)[-5:]
            first_ts = recent_bars[0]['timestamp']
            last_ts = recent_bars[-1]['timestamp']
            last_5m_ts = self.last_5min_bar_timestamps.get(symbol)
            if not can_resample_5min(len(recent_bars), first_ts, last_ts, last_5m_ts):
                return
            five_min_bar = resample_bars_to_5min(recent_bars, symbol)
            self.five_min_buffers[symbol].append(five_min_bar)
            self.last_5min_bar_timestamps[symbol] = last_ts
        self.logger.info(f"{symbol.upper()} 5m | {five_min_bar['timestamp']} | C:{five_min_bar['close']:.4f} V:{five_min_bar['volume']:.2f}")
        if self.on_5min_bar_callback:
            try:
                self.on_5min_bar_callback(five_min_bar)
            except Exception as e:
                self.logger.error(f"Callback error: {e}", exc_info=True)

    def _check_parquet_save(self) -> None:
        current_time = datetime.utcnow()
        if current_time >= self.state.last_parquet_save + timedelta(hours=PARQUET_SAVE_INTERVAL_HOURS):
            self._save_parquet()
            self.state.last_parquet_save = current_time

    def _save_parquet(self) -> None:
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H')
            for symbol in self.symbols:
                self._save_symbol_parquet(symbol, timestamp, '1m', self.one_min_buffers)
                self._save_symbol_parquet(symbol, timestamp, '5m', self.five_min_buffers)
        except Exception as e:
            self.logger.error(f"Parquet save failed: {e}", exc_info=True)

    def _save_symbol_parquet(self, symbol: str, timestamp: str, timeframe: str, buffers: Dict[str, deque]) -> None:
        with self.buffer_lock:
            if symbol not in buffers or not buffers[symbol]:
                return
            df = pd.DataFrame(list(buffers[symbol]))
        file_path = os.path.join(self.parquet_dir, f"{symbol}_{timeframe}_{timestamp}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
        self.logger.info(f"Saved {timeframe} parquet: {file_path}")

    def get_buffers(self) -> Dict[str, Dict[str, List]]:
        with self.buffer_lock:
            return {'one_min': {s: list(b) for s, b in self.one_min_buffers.items()}, 'five_min': {s: list(b) for s, b in self.five_min_buffers.items()}}

    def get_status(self) -> Dict[str, Any]:
        return {'connected': self.state.connected, 'running': self.state.running, 'symbols': [s.upper() for s in self.symbols], 'reconnect_attempts': self.state.reconnect_attempts, 'last_parquet_save': self.state.last_parquet_save, 'last_bar_timestamps': dict(self.state.last_bar_timestamps)}


def generate_sample_buffer():
    base_price = 195.50
    current_time = datetime.utcnow().replace(second=0, microsecond=0)
    one_min_bars = []
    five_min_bars = []
    for i in range(100):
        timestamp = current_time - timedelta(minutes=100-i)
        noise = (hash(str(i)) % 1000) / 10000
        trend = i * 0.0001
        volatility = 0.001 if i % 20 != 0 else 0.003
        open_price = base_price + (i * 0.02) + (noise - 0.005)
        close_price = open_price + trend + (noise - 0.005) + (volatility * (hash(str(i)) % 100 - 50) / 100)
        high_price = max(open_price, close_price) + abs(noise * 2) + volatility
        low_price = min(open_price, close_price) - abs(noise * 2) - volatility
        volume = 1500 + (hash(str(i)) % 2000)
        one_min_bars.append({'timestamp': timestamp, 'open': round(open_price, 4), 'high': round(high_price, 4), 'low': round(low_price, 4), 'close': round(close_price, 4), 'volume': round(volume, 2), 'n_trades': int(volume / 10), 'quote_volume': round(volume * open_price, 2), 'taker_buy_base': round(volume * 0.45, 2), 'taker_buy_quote': round(volume * 0.45 * open_price, 2), 'is_closed': True})
    for i in range(20):
        timestamp = current_time - timedelta(minutes=100-i*5)
        start_idx = i * 5
        end_idx = start_idx + 5
        if end_idx <= len(one_min_bars):
            bars_subset = one_min_bars[start_idx:end_idx]
            five_min_bars.append({'timestamp': timestamp, 'open': bars_subset[0]['open'], 'high': max(b['high'] for b in bars_subset), 'low': min(b['low'] for b in bars_subset), 'close': bars_subset[-1]['close'], 'volume': round(sum(b['volume'] for b in bars_subset), 2), 'n_trades': sum(b['n_trades'] for b in bars_subset), 'quote_volume': round(sum(b['quote_volume'] for b in bars_subset), 2), 'taker_buy_base': round(sum(b['taker_buy_base'] for b in bars_subset), 2), 'taker_buy_quote': round(sum(b['taker_buy_quote'] for b in bars_subset), 2)})
    return one_min_bars, five_min_bars


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        print("Generating sample buffer dump...\n")
        one_min, five_min = generate_sample_buffer()
        print("=" * 80)
        print("SOL-PERP LIVE FEED BUFFER DUMP")
        print("=" * 80)
        print(f"\n1-Minute Buffer (showing last 10 of {len(one_min)} bars):")
        print("-" * 80)
        for bar in one_min[-10:]:
            print(f"{bar['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} {bar['open']:<8.4f} {bar['high']:<8.4f} {bar['low']:<8.4f} {bar['close']:<8.4f} {bar['volume']:<10.2f}")
        print(f"\n5-Minute Resampled Buffer (showing last 5 of {len(five_min)} bars):")
        print("-" * 80)
        for bar in five_min[-5:]:
            print(f"{bar['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} {bar['open']:<8.4f} {bar['high']:<8.4f} {bar['low']:<8.4f} {bar['close']:<8.4f} {bar['volume']:<10.2f}")
    else:
        print("Starting live WebSocket feed handler...")
        feed = BinanceWebSocketFeed(symbols='SOLUSDT')
        feed.start()
        try:
            while True:
                time.sleep(30)
                status = feed.get_status()
                print(f"Status: Connected={status['connected']}, Reconnects={status['reconnect_attempts']}")
        except KeyboardInterrupt:
            print("\nShutting down...")
            feed.stop()
