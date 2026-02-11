# websocket_handler_improved.py
"""
Binance Futures WebSocket Feed Handler - Refactored
Plumbing only: connect, normalize bars, emit bars

5m aggregation: bucket-based (not delta/span).
Each 1m bar is assigned to a 5m bucket via floor_to_5min(timestamp).
When a new bucket starts, the previous bucket is validated and emitted.
"""

import websocket
import json
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import os
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

DEFAULT_BUFFER_SIZE = 100
DEFAULT_CSV_DIR = 'data/live_buffer'
RECONNECT_DELAY_INITIAL_SEC = 5
RECONNECT_DELAY_MAX_SEC = 300
PING_INTERVAL_SEC = 60
PING_TIMEOUT_SEC = 10
CSV_SAVE_INTERVAL_HOURS = 1
REQUIRED_KLINE_FIELDS = ('t', 'o', 'h', 'l', 'c', 'v', 'n', 'q', 'V', 'Q', 'x')


@dataclass
class WebSocketState:
    connected: bool = False
    running: bool = False
    reconnect_attempts: int = 0
    current_reconnect_delay: float = RECONNECT_DELAY_INITIAL_SEC
    last_csv_save: datetime = field(default_factory=datetime.utcnow)
    last_bar_timestamps: Dict[str, Optional[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure functions: message parsing & validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pure functions: 5m bucket aggregation
# ---------------------------------------------------------------------------

def floor_to_5min(ts: datetime) -> datetime:
    """Deterministic bucket key from any 1m bar timestamp."""
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)


def validate_5min_bucket(bars: List[Dict[str, Any]], bucket_start: datetime) -> Tuple[bool, Optional[str]]:
    """
    Invariant enforcement before emission.
    Checks: count == 5, all in same bucket, strictly increasing, no duplicates.
    """
    if len(bars) != 5:
        return False, f"Expected 5 bars, got {len(bars)}"
    timestamps = [b['timestamp'] for b in bars]
    for ts in timestamps:
        if floor_to_5min(ts) != bucket_start:
            return False, f"Bar {ts} outside bucket {bucket_start}"
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            return False, f"Non-increasing: {timestamps[i-1]} -> {timestamps[i]}"
    if len(set(timestamps)) != 5:
        return False, f"Duplicate timestamps: {timestamps}"
    return True, None


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


# ---------------------------------------------------------------------------
# Pure functions: infrastructure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BinanceWebSocketFeed
# ---------------------------------------------------------------------------

class BinanceWebSocketFeed:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        csv_dir: str = DEFAULT_CSV_DIR,
        on_5min_bar_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_1min_bar_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        if symbols is None:
            symbols = ['SOLUSDT']
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = [s.lower() for s in symbols]
        self.buffer_size = buffer_size
        self.csv_dir = csv_dir
        self.on_5min_bar_callback = on_5min_bar_callback
        self.on_1min_bar_callback = on_1min_bar_callback
        self.ws_url = build_stream_url(self.symbols)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.state = WebSocketState(last_bar_timestamps={s: None for s in self.symbols})
        self.one_min_buffers: Dict[str, deque] = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.five_min_buffers: Dict[str, deque] = {s: deque(maxlen=buffer_size) for s in self.symbols}

        # Bucket-based 5m aggregation state
        self._active_buckets: Dict[str, Dict] = {}
        self._last_5m_emit_time: Dict[str, Optional[datetime]] = {s: None for s in self.symbols}

        self.buffer_lock = threading.Lock()
        self.ws_lock = threading.Lock()
        self.logger = setup_feed_logging()
        os.makedirs(self.csv_dir, exist_ok=True)
        self.logger.info(f"Initialized feed handler for {len(self.symbols)} pairs: {', '.join(self.symbols).upper()}")

    # -- Properties --

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
    def last_csv_save(self) -> datetime:
        return self.state.last_csv_save

    @last_csv_save.setter
    def last_csv_save(self, value: datetime) -> None:
        self.state.last_csv_save = value

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

    # -- Lifecycle --

    def start(self) -> None:
        with self.ws_lock:
            if self.ws and self.state.connected:
                self.logger.warning("WebSocket already running")
                return
            self.state.running = True
            self._attempt_startup_aggregation()
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
        self._save_csv()
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

    # -- WebSocket callbacks --

    def _on_open(self, ws) -> None:
        self.logger.info("WebSocket connection established")
        self.state.connected = True
        self.state.current_reconnect_delay = RECONNECT_DELAY_INITIAL_SEC
        self.state.reconnect_attempts = 0
        # Reset active buckets on reconnect to prevent frozen state
        self._active_buckets.clear()
        self.logger.info("Active buckets reset on reconnect")

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

        # 1m callback
        if self.on_1min_bar_callback:
            try:
                self.on_1min_bar_callback(bar)
            except Exception as e:
                self.logger.error(f"1m callback error: {e}", exc_info=True)

        self._try_resample(symbol, bar)
        self._check_5m_health(symbol, bar['timestamp'])
        self._check_csv_save()

    def _run_forever(self) -> None:
        while self.state.running:
            try:
                if self.ws:
                    self.ws.run_forever(ping_interval=PING_INTERVAL_SEC, ping_timeout=PING_TIMEOUT_SEC)
            except Exception as e:
                self.logger.error(f"WebSocket run_forever error: {e}", exc_info=True)
                if self.state.running:
                    time.sleep(RECONNECT_DELAY_INITIAL_SEC)

    # -- 5m Bucket Aggregation --

    def _try_resample(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Bucket-based 5m aggregation. No delta math, no span checks."""
        bucket_start = floor_to_5min(bar['timestamp'])

        with self.buffer_lock:
            if symbol not in self._active_buckets:
                self._active_buckets[symbol] = {'start': bucket_start, 'bars': []}

            active = self._active_buckets[symbol]

            # Same bucket → accumulate with guards
            if bucket_start == active['start']:
                # Guard: out-of-order rejection at insert time
                if active['bars'] and bar['timestamp'] <= active['bars'][-1]['timestamp']:
                    self.logger.error(
                        f"OUT-OF-ORDER {symbol}: got {bar['timestamp']} "
                        f"after {active['bars'][-1]['timestamp']}"
                    )
                    return
                # Guard: gap detection on append
                if active['bars']:
                    delta = bar['timestamp'] - active['bars'][-1]['timestamp']
                    if delta != timedelta(minutes=1):
                        self.logger.error(
                            f"1m GAP {symbol}: {active['bars'][-1]['timestamp']} -> "
                            f"{bar['timestamp']} (delta={delta})"
                        )
                active['bars'].append(bar)
                return

            # New bucket → validate and emit previous
            prev_bars = active['bars']
            prev_start = active['start']
            self._active_buckets[symbol] = {'start': bucket_start, 'bars': [bar]}

        # Outside lock: validate and emit previous bucket
        self._emit_bucket(symbol, prev_bars, prev_start)

    def _emit_bucket(self, symbol: str, bars: List[Dict[str, Any]], bucket_start: datetime) -> None:
        """Validate invariants and emit a completed 5m bar."""
        valid, error = validate_5min_bucket(bars, bucket_start)
        if not valid:
            self.logger.error(
                f"5m INVARIANT FAIL {symbol}: {error} | "
                f"bucket={bucket_start} | "
                f"timestamps={[str(b['timestamp']) for b in bars]}"
            )
            return

        five_min_bar = resample_bars_to_5min(bars, symbol)
        with self.buffer_lock:
            self.five_min_buffers[symbol].append(five_min_bar)

        # Structured emission log (grep-friendly)
        self.logger.info(
            f"{symbol.upper()} 5m EMIT | bucket={bucket_start.strftime('%H:%M')} | "
            f"bars={len(bars)} | first={bars[0]['timestamp'].strftime('%H:%M')} "
            f"last={bars[-1]['timestamp'].strftime('%H:%M')} | "
            f"C:{five_min_bar['close']:.4f} V:{five_min_bar['volume']:.2f}"
        )

        # Health: market-time, not wall-time
        self._last_5m_emit_time[symbol] = five_min_bar['timestamp']

        if self.on_5min_bar_callback:
            try:
                self.on_5min_bar_callback(five_min_bar)
            except Exception as e:
                self.logger.error(f"5m callback error: {e}", exc_info=True)

    def _check_5m_health(self, symbol: str, current_bar_ts: datetime) -> None:
        """CRITICAL log if 1m bars are flowing but 5m has been silent >10 minutes."""
        last_emit = self._last_5m_emit_time.get(symbol)
        if last_emit is None:
            return
        gap = current_bar_ts - last_emit
        if gap > timedelta(minutes=10):
            self.logger.critical(
                f"5m STALE {symbol}: last_emit={last_emit}, "
                f"current_1m={current_bar_ts}, gap={gap}"
            )

    def _attempt_startup_aggregation(self) -> None:
        """On boot: emit any complete closed buckets from pre-existing buffer."""
        for symbol in self.symbols:
            with self.buffer_lock:
                buffer = self.one_min_buffers.get(symbol)
                if not buffer or len(buffer) < 5:
                    continue
                # Group existing bars by bucket
                buckets_map = defaultdict(list)
                for bar in buffer:
                    key = floor_to_5min(bar['timestamp'])
                    buckets_map[key].append(bar)

            if not buckets_map:
                continue

            # Current bucket = latest bar's bucket (do NOT emit — in-progress)
            latest_bar_ts = max(b['timestamp'] for b in buffer)
            current_bucket = floor_to_5min(latest_bar_ts)

            emitted = 0
            for bucket_start in sorted(buckets_map.keys()):
                if bucket_start == current_bucket:
                    continue  # skip in-progress bucket
                self._emit_bucket(symbol, buckets_map[bucket_start], bucket_start)
                emitted += 1

            if emitted > 0:
                self.logger.info(f"Startup aggregation: emitted {emitted} closed bucket(s) for {symbol}")

    # -- CSV Persistence (append-safe) --

    def _check_csv_save(self) -> None:
        current_time = datetime.utcnow()
        if current_time >= self.state.last_csv_save + timedelta(hours=CSV_SAVE_INTERVAL_HOURS):
            self._save_csv()
            self.state.last_csv_save = current_time

    def _save_csv(self) -> None:
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H')
            for symbol in self.symbols:
                self._save_symbol_csv(symbol, timestamp, '1m', self.one_min_buffers)
                self._save_symbol_csv(symbol, timestamp, '5m', self.five_min_buffers)
        except Exception as e:
            self.logger.error(f"CSV save failed: {e}", exc_info=True)

    def _save_symbol_csv(self, symbol: str, timestamp: str, timeframe: str, buffers: Dict[str, deque]) -> None:
        with self.buffer_lock:
            if symbol not in buffers or not buffers[symbol]:
                return
            df = pd.DataFrame(list(buffers[symbol]))
        file_path = os.path.join(self.csv_dir, f"{symbol}_{timeframe}_{timestamp}.csv")
        write_header = not os.path.exists(file_path)
        df.to_csv(file_path, index=False, mode='a', header=write_header)
        self.logger.info(f"Saved {timeframe} csv: {file_path}")

    # -- Status --

    def get_buffers(self) -> Dict[str, Dict[str, List]]:
        with self.buffer_lock:
            return {'one_min': {s: list(b) for s, b in self.one_min_buffers.items()}, 'five_min': {s: list(b) for s, b in self.five_min_buffers.items()}}

    def get_status(self) -> Dict[str, Any]:
        return {
            'connected': self.state.connected,
            'running': self.state.running,
            'symbols': [s.upper() for s in self.symbols],
            'reconnect_attempts': self.state.reconnect_attempts,
            'last_csv_save': self.state.last_csv_save,
            'last_bar_timestamps': dict(self.state.last_bar_timestamps),
            'active_buckets': {s: str(b.get('start')) for s, b in self._active_buckets.items()},
            'last_5m_emit': {s: str(t) for s, t in self._last_5m_emit_time.items()}
        }


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        print("Generating sample buffer dump...\n")
        base_price = 195.50
        current_time = datetime.utcnow().replace(second=0, microsecond=0)
        one_min = []
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
            one_min.append({'timestamp': timestamp, 'open': round(open_price, 4), 'high': round(high_price, 4), 'low': round(low_price, 4), 'close': round(close_price, 4), 'volume': round(volume, 2), 'n_trades': int(volume / 10), 'quote_volume': round(volume * open_price, 2), 'taker_buy_base': round(volume * 0.45, 2), 'taker_buy_quote': round(volume * 0.45 * open_price, 2), 'is_closed': True})
        print("=" * 80)
        print("SOL-PERP LIVE FEED BUFFER DUMP")
        print("=" * 80)
        print(f"\n1-Minute Buffer (showing last 10 of {len(one_min)} bars):")
        print("-" * 80)
        for bar in one_min[-10:]:
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
