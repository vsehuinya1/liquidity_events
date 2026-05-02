# websocket_handler_improved.py
"""
Binance Futures WebSocket Feed Handler - v2.1.0
Plumbing only: connect, normalize bars, emit bars

v2.1.0: REST API polling fallback for futures klines when WS is dry
v2.0.0: Race-free reconnection, watchdog, telegram alerts
- Structured lifecycle logging

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
import sys
import logging
import asyncio
import urllib.request
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

# Watchdog thresholds
STALE_RECONNECT_SEC = 300     # Force reconnect if no data for 5 minutes
STALE_FORCE_EXIT_SEC = 1200    # Force process exit if no data for 20 minutes
WATCHDOG_CHECK_INTERVAL_SEC = 30

# Max reconnect attempts before full client re-init
MAX_RECONNECT_BEFORE_REINIT = 10

# v2.1.0: REST polling fallback
REST_POLL_INTERVAL_SEC = 60       # Poll every 60s
REST_STALE_THRESHOLD_SEC = 90     # Activate REST after 90s of no WS data
REST_API_BASE = 'https://fapi.binance.com'  # Futures REST (confirmed working)


@dataclass
class WebSocketState:
    connected: bool = False
    running: bool = False
    reconnect_attempts: int = 0
    total_reconnects: int = 0
    msg_count: int = 0
    current_reconnect_delay: float = RECONNECT_DELAY_INITIAL_SEC
    last_csv_save: datetime = field(default_factory=datetime.utcnow)
    last_bar_timestamps: Dict[str, Optional[int]] = field(default_factory=dict)
    last_message_time: float = 0.0       # wall-clock of last message (WS or REST)
    last_ws_message_time: float = 0.0    # v2.1.0: wall-clock of last WS-only message


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


# ---------------------------------------------------------------------------
# Pure functions: REST API polling (v2.1.0)
# ---------------------------------------------------------------------------

def fetch_rest_klines(symbol: str, interval: str = '1m', limit: int = 2) -> Optional[List[Dict[str, Any]]]:
    """Fetch closed klines from Binance Futures REST API. Returns list of bar dicts or None on error."""
    url = f"{REST_API_BASE}/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        bars = []
        for k in data:
            # Binance REST kline format: [open_time, open, high, low, close, volume, close_time, ...]
            # Only process closed bars (close_time < now)
            close_time_ms = int(k[6])
            if close_time_ms > time.time() * 1000:
                continue  # Bar not yet closed
            bars.append({
                'symbol': symbol.upper(),
                'timestamp': pd.to_datetime(int(k[0]), unit='ms'),
                'timestamp_ms': int(k[0]),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'n_trades': int(k[8]),
                'quote_volume': float(k[7]),
                'taker_buy_base': float(k[9]),
                'taker_buy_quote': float(k[10]),
                'is_closed': True
            })
        return bars
    except Exception:
        return None


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
# BinanceWebSocketFeed — v2.0.0 (race-free lifecycle)
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
        self.logger = setup_feed_logging()
        os.makedirs(self.csv_dir, exist_ok=True)

        # v2.0: Thread lifecycle management
        self._ws_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._rest_poller_thread: Optional[threading.Thread] = None  # v2.1.0
        self._rest_active: bool = False  # v2.1.0: True when REST is feeding data

        # v2.0: Telegram alerting (set via set_telegram())
        self._telegram = None
        self._async_loop = None

        self.logger.info(f"Initialized feed handler v2.1 for {len(self.symbols)} pairs: {', '.join(self.symbols).upper()}")

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

    # -- Telegram Integration (v2.0) --

    def set_telegram(self, telegram_bot, async_loop) -> None:
        """Inject Telegram bot and event loop for feed lifecycle alerts."""
        self._telegram = telegram_bot
        self._async_loop = async_loop
        self.logger.info("Telegram alerting configured for feed handler")

    def _send_telegram_alert(self, message: str) -> None:
        """Send feed lifecycle alert via Telegram. Thread-safe, non-blocking."""
        if not self._telegram or not self._async_loop:
            return
        try:
            if self._async_loop.is_running():
                async def _send():
                    try:
                        await self._telegram.send_message(
                            f"🔧 <b>Feed Alert</b>\n{message}"
                        )
                    except Exception as e:
                        self.logger.error(f"Telegram feed alert failed: {e}")
                asyncio.run_coroutine_threadsafe(_send(), self._async_loop)
        except Exception as e:
            self.logger.error(f"Telegram alert scheduling failed: {e}")

    # -- Lifecycle (v2.0: race-free) --

    def start(self) -> None:
        """
        Start the WebSocket feed.

        v2.0: Guarantees:
        - Only one _run_forever thread exists at any time
        - Old thread is joined before new one starts
        - Watchdog thread monitors data freshness
        """
        if self.state.running and self._ws_thread and self._ws_thread.is_alive():
            self.logger.warning("Feed handler already running")
            return

        self.state.running = True
        self.state.last_message_time = time.time()  # Initialize freshness timer
        self.state.last_ws_message_time = time.time()  # Initialize WS freshness

        self._attempt_startup_aggregation()

        # Build initial WebSocketApp
        self._build_ws()

        # Start the single WS lifecycle thread
        self._ws_thread = threading.Thread(
            target=self._run_forever_loop,
            daemon=True,
            name="ws-lifecycle"
        )
        self._ws_thread.start()
        self.logger.info(f"[LIFECYCLE] ws-lifecycle thread started (tid={self._ws_thread.ident})")

        # Start watchdog thread
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="ws-watchdog"
            )
            self._watchdog_thread.start()
            self.logger.info(f"[LIFECYCLE] ws-watchdog thread started (tid={self._watchdog_thread.ident})")

        # v2.1.0: Start REST poller thread
        if self._rest_poller_thread is None or not self._rest_poller_thread.is_alive():
            self._rest_poller_thread = threading.Thread(
                target=self._rest_poller_loop,
                daemon=True,
                name="ws-rest-poller"
            )
            self._rest_poller_thread.start()
            self.logger.info(f"[LIFECYCLE] REST poller thread started (tid={self._rest_poller_thread.ident})")

        self.logger.info(f"Feed handler started for {len(self.symbols)} pairs")

    def stop(self) -> None:
        """
        Stop the WebSocket feed cleanly.

        v2.0: Ensures the WS thread is fully terminated before returning.
        """
        self.logger.info("[LIFECYCLE] Stopping feed handler...")
        self.state.running = False
        self.state.connected = False

        # Close the WebSocket connection to unblock run_forever()
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        # Wait for the WS thread to finish
        if self._ws_thread and self._ws_thread.is_alive():
            self.logger.info("[LIFECYCLE] Waiting for ws-lifecycle thread to exit...")
            self._ws_thread.join(timeout=15)
            if self._ws_thread.is_alive():
                self.logger.warning("[LIFECYCLE] ws-lifecycle thread did not exit cleanly")
            else:
                self.logger.info("[LIFECYCLE] ws-lifecycle thread exited")

        self._save_csv()
        self.logger.info("[LIFECYCLE] Feed handler stopped")

    def _build_ws(self) -> None:
        """Create a fresh WebSocketApp instance. No threads spawned."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

    # -- WebSocket callbacks (v2.0: NO reconnection from callbacks) --

    def _on_open(self, ws) -> None:
        self.logger.info("[WS] Connection established")
        self.state.msg_count += 1
        payload = {
            "method": "SUBSCRIBE",
            "params": [f"{s.lower()}@kline_1m" for s in self.symbols],
            "id": self.state.msg_count
        }
        ws.send(json.dumps(payload))
        self.state.connected = True
        self.state.current_reconnect_delay = RECONNECT_DELAY_INITIAL_SEC
        self.state.last_message_time = time.time()
        # Reset active buckets on reconnect to prevent frozen state
        self._active_buckets.clear()
        self.logger.info("[WS] Active buckets reset on connect")

        if self.state.reconnect_attempts > 0:
            self.state.total_reconnects += 1
            self._send_telegram_alert(
                f"✅ Feed restored after {self.state.reconnect_attempts} attempt(s)\n"
                f"Total reconnects: {self.state.total_reconnects}"
            )
            self.state.reconnect_attempts = 0

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """
        v2.0: Only log and update state. Do NOT reconnect here.
        _run_forever_loop handles reconnection after ws.run_forever() returns.
        """
        self.logger.warning(f"[WS] Connection closed: {close_status_code} - {close_msg}")
        self.state.connected = False

    def _on_error(self, ws, error) -> None:
        """
        v2.0: Only log and update state. Do NOT reconnect here.
        _run_forever_loop handles reconnection after ws.run_forever() returns.
        """
        self.logger.error(f"[WS] Error: {error}")
        self.state.connected = False

    def _on_message(self, ws, message: str) -> None:
        # Update overall freshness on every WS message
        self.state.last_message_time = time.time()
        # NOTE: last_ws_message_time is only updated below when we get actual kline data

        payload = parse_raw_message(message)
        if payload is None:
            self.logger.error(f"JSON decode error. Raw: {message[:200]}")
            return
        
        kline = extract_kline_from_payload(payload)
        if kline is None:
            if 'result' in payload:
                self.logger.info(f"Subscription confirmed: {payload}")
            return
        # Got actual kline data — WS is truly alive
        self.state.last_ws_message_time = time.time()
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

    # -- v2.0: Single-threaded lifecycle loop (replaces _run_forever + reconnect) --

    def _run_forever_loop(self) -> None:
        """
        Single-threaded WebSocket lifecycle loop.

        v2.0 GUARANTEES:
        - This is the ONLY thread that calls ws.run_forever()
        - Reconnection happens IN this thread after run_forever() returns
        - No parallel _run_forever threads can exist
        - Exponential backoff with max delay
        - After MAX_RECONNECT_BEFORE_REINIT failures, rebuild the entire client
        """
        self.logger.info("[LIFECYCLE] _run_forever_loop entered")

        while self.state.running:
            try:
                if self.ws:
                    self.logger.info("[LIFECYCLE] Calling ws.run_forever()")
                    self.ws.run_forever(
                        ping_interval=PING_INTERVAL_SEC,
                        ping_timeout=PING_TIMEOUT_SEC
                    )
                    self.logger.info("[LIFECYCLE] ws.run_forever() returned")
            except Exception as e:
                self.logger.error(f"[LIFECYCLE] ws.run_forever() exception: {e}", exc_info=True)

            # If we're shutting down, don't reconnect
            if not self.state.running:
                self.logger.info("[LIFECYCLE] Shutdown flag set, exiting loop")
                break

            # -- Reconnection logic (all within this single thread) --
            self.state.connected = False
            self.state.reconnect_attempts += 1
            delay = self.state.current_reconnect_delay

            self.logger.warning(
                f"[RECONNECT] Attempt #{self.state.reconnect_attempts} | "
                f"Delay: {delay:.0f}s | "
                f"Total reconnects: {self.state.total_reconnects}"
            )

            self._send_telegram_alert(
                f"🔌 Feed disconnected\n"
                f"Reconnect attempt #{self.state.reconnect_attempts}\n"
                f"Retrying in {delay:.0f}s..."
            )

            # Sleep with interruptible check
            sleep_end = time.time() + delay
            while time.time() < sleep_end and self.state.running:
                time.sleep(1)

            if not self.state.running:
                break

            # Exponential backoff
            self.state.current_reconnect_delay = min(
                delay * 2,
                RECONNECT_DELAY_MAX_SEC
            )

            # After too many failures, do a full client re-init
            if self.state.reconnect_attempts >= MAX_RECONNECT_BEFORE_REINIT:
                self.logger.warning(
                    f"[RECONNECT] {self.state.reconnect_attempts} failures, "
                    f"rebuilding WebSocket client from scratch"
                )
                self._send_telegram_alert(
                    f"⚠️ {self.state.reconnect_attempts} reconnect failures\n"
                    f"Rebuilding WS client from scratch..."
                )
                # Rebuild the URL in case of DNS/routing issues
                self.ws_url = build_stream_url(self.symbols)

            # Build fresh WebSocketApp for next iteration
            self._build_ws()
            self.logger.info("[RECONNECT] Fresh WebSocketApp created, re-entering run_forever()")

        self.logger.info("[LIFECYCLE] _run_forever_loop exited")

    # -- v2.0: Data Freshness Watchdog --

    def _watchdog_loop(self) -> None:
        """
        Watchdog thread: monitors data freshness.

        - If no data for STALE_RECONNECT_SEC: force close WS to trigger reconnect
        - If no data for STALE_FORCE_EXIT_SEC: kill the process (let PM2 restart)
        """
        self.logger.info("[WATCHDOG] Started")

        while self.state.running:
            time.sleep(WATCHDOG_CHECK_INTERVAL_SEC)

            if not self.state.running:
                break

            elapsed = time.time() - self.state.last_message_time

            # Level 1: Force reconnect
            if elapsed > STALE_RECONNECT_SEC and self.state.connected:
                self.logger.warning(
                    f"[WATCHDOG] Data stale for {elapsed:.0f}s (>{STALE_RECONNECT_SEC}s). "
                    f"Forcing WebSocket close to trigger reconnect."
                )
                self._send_telegram_alert(
                    f"⚠️ No data for {elapsed:.0f}s\n"
                    f"Forcing reconnect..."
                )
                # Close the WS — this will cause run_forever() to return,
                # which triggers the reconnect logic in _run_forever_loop
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception:
                        pass
                self.state.connected = False

            # Level 2: Force process exit (fail fast — let PM2/supervisor restart)
            elif elapsed > STALE_FORCE_EXIT_SEC:
                self.logger.critical(
                    f"[WATCHDOG] CRITICAL: Data stale for {elapsed:.0f}s "
                    f"(>{STALE_FORCE_EXIT_SEC}s). FORCING PROCESS EXIT."
                )
                self._send_telegram_alert(
                    f"🔴 Feed dead for {elapsed:.0f}s\n"
                    f"Forcing process restart..."
                )
                # Give Telegram a moment to send
                time.sleep(2)
                # Hard exit — PM2 will restart the process
                os._exit(1)

        self.logger.info("[WATCHDOG] Stopped")

    # -- v2.0: Force reconnect (called by watchdog) --

    def _force_reconnect(self) -> None:
        """Force a reconnect by closing the current WS connection."""
        self.logger.warning("[FORCE_RECONNECT] Closing WS to trigger reconnect")
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.state.connected = False

    # -- v2.1.0: REST API Polling Fallback --

    def _rest_poller_loop(self) -> None:
        """
        REST polling fallback thread (v2.1.0).
        When WS data is stale for >REST_STALE_THRESHOLD_SEC, polls fapi.binance.com
        for closed 1m klines and feeds them through the normal bar pipeline.
        Deduplicates against already-seen bar timestamps.
        """
        self.logger.info("[REST_POLLER] Started")

        while self.state.running:
            time.sleep(REST_POLL_INTERVAL_SEC)

            if not self.state.running:
                break

            elapsed_ws = time.time() - self.state.last_ws_message_time

            # Only activate when WS is dry (use WS-specific timer)
            if elapsed_ws < REST_STALE_THRESHOLD_SEC:
                if self._rest_active:
                    self._rest_active = False
                    self.logger.info("[REST_POLLER] WS data resumed, REST polling deactivated")
                    self._send_telegram_alert("✅ WS data resumed, REST fallback deactivated")
                continue

            # WS is stale — activate REST polling
            if not self._rest_active:
                self._rest_active = True
                self.logger.warning(f"[REST_POLLER] WS stale for {elapsed_ws:.0f}s, activating REST fallback")
                self._send_telegram_alert(
                    f"⚠️ WS stale for {elapsed_ws:.0f}s\n"
                    f"REST polling fallback activated (fapi.binance.com)"
                )

            # Poll each symbol
            for symbol in self.symbols:
                bars = fetch_rest_klines(symbol, limit=3)
                if bars is None:
                    self.logger.error(f"[REST_POLLER] Failed to fetch klines for {symbol}")
                    continue

                for bar in bars:
                    bar_ts_ms = bar['timestamp_ms']
                    last_ts_ms = self.state.last_bar_timestamps.get(symbol)

                    # Dedup: skip if already seen
                    if not is_new_bar(bar_ts_ms, last_ts_ms):
                        continue

                    # Validate
                    valid, error = validate_bar_integrity(
                        bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                    )
                    if not valid:
                        self.logger.warning(f"[REST_POLLER] Bar integrity fail {symbol}: {error}")
                        continue

                    # Process bar through normal pipeline
                    self.state.last_bar_timestamps[symbol] = bar_ts_ms
                    self.state.last_message_time = time.time()  # Reset overall freshness (not WS)

                    with self.buffer_lock:
                        self.one_min_buffers[symbol].append(bar)

                    self.logger.info(
                        f"{symbol.upper()} 1m [REST] | {bar['timestamp']} | "
                        f"C:{bar['close']:.4f} V:{bar['volume']:.2f}"
                    )

                    # 1m callback
                    if self.on_1min_bar_callback:
                        try:
                            self.on_1min_bar_callback(bar)
                        except Exception as e:
                            self.logger.error(f"[REST_POLLER] 1m callback error: {e}", exc_info=True)

                    self._try_resample(symbol, bar)
                    self._check_5m_health(symbol, bar['timestamp'])

        self.logger.info("[REST_POLLER] Stopped")

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
        elapsed = time.time() - self.state.last_message_time if self.state.last_message_time > 0 else -1
        return {
            'connected': self.state.connected,
            'running': self.state.running,
            'symbols': [s.upper() for s in self.symbols],
            'reconnect_attempts': self.state.reconnect_attempts,
            'total_reconnects': self.state.total_reconnects,
            'last_csv_save': self.state.last_csv_save,
            'last_bar_timestamps': dict(self.state.last_bar_timestamps),
            'active_buckets': {s: str(b.get('start')) for s, b in self._active_buckets.items()},
            'last_5m_emit': {s: str(t) for s, t in self._last_5m_emit_time.items()},
            'data_age_sec': round(elapsed, 1)
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
        print("Starting live WebSocket feed handler v2.0...")
        feed = BinanceWebSocketFeed(symbols='SOLUSDT')
        feed.start()
        try:
            while True:
                time.sleep(30)
                status = feed.get_status()
                print(f"Status: Connected={status['connected']}, Reconnects={status['total_reconnects']}, DataAge={status['data_age_sec']}s")
        except KeyboardInterrupt:
            print("\nShutting down...")
            feed.stop()
