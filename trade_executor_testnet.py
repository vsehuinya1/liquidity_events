# trade_executor_testnet.py
"""
Testnet Trade Executor for Verification
v2.1.0: Correctness fixes for partial fills, rate limiting, convergent exits

Exit Detection Modes:
1. USER_DATA_STREAM: Real-time via ORDER_TRADE_UPDATE (primary, <1s latency)
2. ORDER_POLL: Periodic via futures_get_order (secondary, 5s interval)
3. POSITION_POLL: Fallback via position information (tertiary)

v2.1 Fixes:
- Cumulative partial-fill aggregation (convergent exit)
- Rate-limit emergency fallback (MARKET close)
- Add-only reconciliation (no zombie positions)
- Exchange timestamps (no clock skew issues)
"""

import logging
import asyncio
import json
import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Set
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

from bot_config import BINANCE_API_KEY as BINANCE_TESTNET_API_KEY, BINANCE_API_SECRET as BINANCE_TESTNET_SECRET

STATE_FILE_PATH = 'data/state/execution_state.json'
KILL_FLAG_FILE_PATH = 'data/state/kill_switch.flag'
METRIC_LOG_PATH = 'logs/verify_latency.csv'
METRIC_CSV_HEADER = "Time_Signal,Time_Sent,Time_Ack,Latency_Int_ms,Latency_Net_ms,Symbol,Direction,Size_Mult,Expected_Px,Fill_Px,Slippage_Bp,Attack_Mode,Bar_Range\n"
BASE_SIZE_USD = 100
LATENCY_THRESHOLD_MS = 3000
TRAILING_STOP_MULTIPLIER = 1.8
HARD_STOP_DISTANCE_PERCENT = 0.10
TRAILING_LOOP_INTERVAL_SEC = 5
RECONCILIATION_INTERVAL_SEC = 60
RATE_LIMIT_RETRY_MS = 500
RATE_LIMIT_MAX_RETRIES = 3
PARTIAL_FILL_EPSILON = 0.001  # Tolerance for "fully filled" check


@dataclass
class ExecutorState:
    kill_switch_active: bool = False
    active_orders: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PartialFillTracker:
    """Track cumulative fills for an order to handle partial fills correctly."""
    order_id: int
    position_qty: float
    filled_qty: float = 0.0
    fill_ids_seen: Set[int] = field(default_factory=set)
    vwap_numerator: float = 0.0  # sum(price * qty)
    
    def add_fill(self, trade_id: int, fill_qty: float, fill_price: float) -> bool:
        """Add a fill. Returns True if this is a new fill, False if duplicate."""
        if trade_id in self.fill_ids_seen:
            return False  # Duplicate fill event
        self.fill_ids_seen.add(trade_id)
        self.filled_qty += fill_qty
        self.vwap_numerator += fill_price * fill_qty
        return True
    
    def is_complete(self) -> bool:
        """Returns True if position is fully closed."""
        return self.filled_qty >= (self.position_qty - PARTIAL_FILL_EPSILON)
    
    def get_vwap(self) -> float:
        """Returns volume-weighted average price of all fills."""
        if self.filled_qty == 0:
            return 0.0
        return self.vwap_numerator / self.filled_qty


def validate_credentials() -> None:
    if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_SECRET:
        raise ValueError("Missing Binance Testnet Credentials")


def check_persistent_kill_flag(kill_flag_path: str) -> bool:
    return os.path.exists(kill_flag_path)


def should_block_execution(state: ExecutorState) -> bool:
    return state.kill_switch_active


def parse_signal_timestamp(timestamp: Any) -> float:
    if isinstance(timestamp, str):
        ts_obj = pd.to_datetime(timestamp)
        return ts_obj.timestamp()
    return timestamp.timestamp()


def compute_latency_ms(t_received: float, t_signal: float) -> float:
    return (t_received - t_signal) * 1000


def is_latency_exceeded(latency_ms: float, threshold_ms: float = LATENCY_THRESHOLD_MS) -> bool:
    return latency_ms > threshold_ms


def compute_quantity(base_size_usd: float, size_mult: float, entry_price: float) -> float:
    quantity = round((base_size_usd * size_mult) / entry_price, 0)
    return max(quantity, 1)


def compute_hard_stop_price(fill_price: float, direction: str, distance_pct: float = HARD_STOP_DISTANCE_PERCENT) -> float:
    hard_dist = fill_price * distance_pct
    if direction == 'LONG':
        return round(fill_price - hard_dist, 2)
    return round(fill_price + hard_dist, 2)


def compute_trailing_stop(current_price: float, current_stop: float, atr: float, direction: str, multiplier: float = TRAILING_STOP_MULTIPLIER) -> tuple:
    if direction == 'LONG':
        proposed = current_price - (multiplier * atr)
        if proposed > current_stop:
            return proposed, True
    else:
        proposed = current_price + (multiplier * atr)
        if proposed < current_stop:
            return proposed, True
    return current_stop, False


def compute_slippage_bp(fill_price: float, expected_price: float) -> float:
    return abs((fill_price - expected_price) / expected_price) * 10000


def format_metric_row(signal_timestamp, t_sent, t_ack, latency_int_ms, latency_net_ms, symbol, direction, size_mult, expected_price, fill_price, slippage_bp, is_attack, bar_range) -> str:
    return f"{signal_timestamp},{t_sent},{t_ack},{latency_int_ms:.2f},{latency_net_ms:.2f},{symbol},{direction},{size_mult},{expected_price},{fill_price},{slippage_bp:.2f},{is_attack},{bar_range:.4f}\n"


def load_state_from_file(state_file: str) -> Dict:
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state_to_file(state_file: str, active_orders: Dict) -> None:
    with open(state_file, 'w') as f:
        json.dump(active_orders, f, indent=2)


def write_kill_flag(kill_flag_path: str) -> None:
    with open(kill_flag_path, 'w') as f:
        f.write(f"KILLED_AT_{datetime.utcnow().isoformat()}")


def append_metric_log(metric_log_path: str, row: str) -> None:
    with open(metric_log_path, 'a') as f:
        f.write(row)


def ensure_metric_log_exists(metric_log_path: str) -> None:
    if not os.path.exists(metric_log_path):
        with open(metric_log_path, 'w') as f:
            f.write(METRIC_CSV_HEADER)


def setup_executor_logging() -> logging.Logger:
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('TestnetExecutor')
    logger.setLevel(logging.INFO)
    return logger


def generate_trade_id() -> str:
    return str(uuid.uuid4())[:8]


class TestnetTradeExecutor:
    def __init__(self, telegram_bot=None, orchestrator=None):
        validate_credentials()
        self.client = Client(BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, testnet=True)
        self.telegram = telegram_bot
        self.orchestrator = orchestrator
        self.logger = setup_executor_logging()
        self.state_file = STATE_FILE_PATH
        self.kill_flag_file = KILL_FLAG_FILE_PATH
        self.metric_log_path = METRIC_LOG_PATH
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        ensure_metric_log_exists(self.metric_log_path)
        self.state = ExecutorState(
            kill_switch_active=check_persistent_kill_flag(self.kill_flag_file),
            active_orders=load_state_from_file(self.state_file)
        )
        self.detectors: Dict[str, Any] = {}
        
        # Event-driven architecture
        self._entry_lock = asyncio.Lock()
        self._exit_lock = threading.Lock()
        
        # v2.1: Partial fill tracking (in-memory, keyed by order_id)
        self._partial_fill_trackers: Dict[int, PartialFillTracker] = {}
        
        # v2.1: Track recently exited symbols to prevent recon resurrection
        self._recently_exited: Dict[str, float] = {}  # symbol -> exit_timestamp
        self._recently_exited_ttl = 10.0  # seconds
        
        if self.state.kill_switch_active:
            self.logger.critical("🚨 STARTUP BLOCKED: 'kill_switch.flag' DETECTED. SYSTEM LOCKED.")
        else:
            self.logger.info("TESTNET Executor v2.1 Initialized. Correctness fixes active.")

    @property
    def KILL_SWITCH(self) -> bool:
        return self.state.kill_switch_active
    
    @KILL_SWITCH.setter
    def KILL_SWITCH(self, value: bool) -> None:
        self.state.kill_switch_active = value

    @property
    def active_orders(self) -> Dict:
        return self.state.active_orders
    
    @active_orders.setter
    def active_orders(self, value: Dict) -> None:
        self.state.active_orders = value

    def register_detector(self, symbol: str, detector_instance) -> None:
        self.detectors[symbol] = detector_instance

    # =========================================================================
    # v2.1 FIX: ADD-ONLY RECONCILIATION
    # =========================================================================
    async def reconcile_with_exchange(self) -> None:
        """
        Reconcile local state with exchange positions.
        v2.1: ADD-ONLY - reconciliation can only ADD positions, never remove.
        Exit events are authoritative for removal.
        """
        try:
            positions = self.client.futures_position_information()
            exchange_positions = {
                p['symbol']: {
                    'positionAmt': float(p['positionAmt']),
                    'entryPrice': float(p['entryPrice']),
                    'updateTime': int(p['updateTime'])
                }
                for p in positions if float(p['positionAmt']) != 0
            }
            
            # v2.1: Clean up recently_exited (TTL expired entries)
            current_time = time.time()
            self._recently_exited = {
                sym: ts for sym, ts in self._recently_exited.items()
                if current_time - ts < self._recently_exited_ttl
            }
            
            added_count = 0
            # ADD-ONLY: Only add positions we don't know about
            for symbol, pos_data in exchange_positions.items():
                if symbol not in self.state.active_orders:
                    # v2.1: Don't resurrect recently exited positions
                    if symbol in self._recently_exited:
                        self.logger.debug(f"🔄 Recon: Skipping {symbol} (recently exited)")
                        continue
                    
                    # Unknown position - add to tracking
                    self.logger.warning(f"🔄 Recon: Discovered untracked position {symbol} on exchange")
                    self.state.active_orders[symbol] = {
                        'trade_id': f"RECON_{generate_trade_id()}",
                        'direction': 'LONG' if pos_data['positionAmt'] > 0 else 'SHORT',
                        'entry_price': pos_data['entryPrice'],
                        'position_qty': abs(pos_data['positionAmt']),
                        'exchange_update_time': pos_data['updateTime'],
                        'discovered_at': current_time
                    }
                    added_count += 1
            
            if added_count > 0:
                save_state_to_file(self.state_file, self.state.active_orders)
            
            self.logger.info(f"🔄 Reconciled: {len(exchange_positions)} on exchange, {len(self.state.active_orders)} tracked, {added_count} added (ADD-ONLY mode)")
        except Exception as e:
            self.logger.error(f"Reconciliation failed: {e}")

    async def periodic_reconciliation(self) -> None:
        """Run reconciliation every RECONCILIATION_INTERVAL_SEC seconds."""
        while True:
            await asyncio.sleep(RECONCILIATION_INTERVAL_SEC)
            if should_block_execution(self.state):
                continue
            await self.reconcile_with_exchange()

    # =========================================================================
    # v2.1 FIX: CONVERGENT PARTIAL-FILL HANDLER
    # =========================================================================
    async def on_order_update(self, order_info: Dict[str, Any]) -> None:
        """
        Handle real-time order fill from userDataStream.
        
        v2.1: CONVERGENT - handles partial fills correctly.
        Any sequence of ORDER_TRADE_UPDATE events results in exactly one
        terminal "position closed" state.
        
        Event fields:
            s: symbol, i: orderId, X: status, t: tradeId (unique per fill)
            ap: avgPrice, l: lastFilledQty, z: cumFilledQty, S: side
            T: transaction time (exchange timestamp)
        """
        order_id = order_info.get('i')     # orderId
        status = order_info.get('X')       # Order status
        symbol = order_info.get('s')       # Symbol
        trade_id = order_info.get('t')     # tradeId (unique per fill)
        last_fill_qty = float(order_info.get('l', 0))  # lastFilledQty
        last_fill_price = float(order_info.get('L', 0))  # lastFilledPrice
        cum_fill_qty = float(order_info.get('z', 0))  # cumFilledQty
        exchange_time = int(order_info.get('T', 0))  # Transaction time (ms)
        
        # Only process FILLED or PARTIALLY_FILLED
        if status not in ['FILLED', 'PARTIALLY_FILLED']:
            return
        
        with self._exit_lock:
            if symbol not in self.state.active_orders:
                self.logger.debug(f"UserData: Ignoring {symbol} (not in active_orders)")
                return
            
            position = self.state.active_orders[symbol]
            sl_order_id = position.get('sl_order_id')
            hard_stop_id = position.get('hard_stop_id')
            
            # Check if this is one of our stop orders
            if order_id not in [sl_order_id, hard_stop_id]:
                self.logger.debug(f"UserData: Order {order_id} not our stop for {symbol}")
                return
            
            position_qty = position.get('position_qty', 0)
            
            # v2.1: Get or create partial fill tracker
            if order_id not in self._partial_fill_trackers:
                self._partial_fill_trackers[order_id] = PartialFillTracker(
                    order_id=order_id,
                    position_qty=position_qty
                )
            
            tracker = self._partial_fill_trackers[order_id]
            
            # Add this fill (dedupe by trade_id)
            is_new_fill = tracker.add_fill(trade_id, last_fill_qty, last_fill_price)
            if not is_new_fill:
                self.logger.debug(f"UserData: Duplicate fill {trade_id} for order {order_id}")
                return
            
            self.logger.info(f"UserData: Fill {trade_id} | {symbol} | qty={last_fill_qty} @ {last_fill_price} | cum={tracker.filled_qty}/{position_qty}")
            
            # v2.1: CONVERGENT - only finalize when fully filled
            if not tracker.is_complete():
                self.logger.info(f"UserData: Partial fill, waiting for more ({tracker.filled_qty:.4f}/{position_qty:.4f})")
                return  # Wait for more fills
            
            # FULLY FILLED - finalize exit
            exit_price = tracker.get_vwap()
            exit_via = 'USER_DATA_STREAM'
            
            # v2.1: Mark as recently exited BEFORE removing from active_orders
            self._recently_exited[symbol] = time.time()
            
            # Remove from active_orders
            del self.state.active_orders[symbol]
            save_state_to_file(self.state_file, self.state.active_orders)
            
            # Cleanup tracker
            del self._partial_fill_trackers[order_id]
        
        # Process exit outside the lock
        await self._handle_exit(
            symbol=symbol,
            position=position,
            exit_price=exit_price,
            exit_via=exit_via,
            exchange_time=exchange_time
        )

    # =========================================================================
    # CENTRALIZED EXIT HANDLER
    # =========================================================================
    async def _handle_exit(
        self, 
        symbol: str, 
        position: Dict[str, Any], 
        exit_price: float, 
        exit_via: str,
        exchange_time: int = 0
    ) -> None:
        """Handle trade exit with notifications and logging."""
        trade_id = position.get('trade_id', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)
        direction = position.get('direction', 'LONG')
        atr = position.get('atr', 1)
        
        # Calculate PnL in R
        if direction == 'LONG':
            pnl_r = (exit_price - entry_price) / (atr * 1.5)
        else:
            pnl_r = (entry_price - exit_price) / (atr * 1.5)
        
        # Telegram notification with retry
        if self.telegram:
            for attempt in range(3):
                try:
                    await self.telegram.send_exit_alert(
                        pair=symbol,
                        direction=direction,
                        exit_price=exit_price,
                        pnl_r=pnl_r,
                        timestamp=datetime.now()
                    )
                    break
                except Exception as e:
                    self.logger.warning(f"Exit alert attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)
        
        # v2.1: Use exchange time if available
        exit_time = datetime.fromtimestamp(exchange_time / 1000) if exchange_time else datetime.now()
        
        self.logger.info(f"EXIT [{trade_id}]: {symbol} {direction} | Entry: {entry_price:.2f} Exit: {exit_price:.4f} | R: {pnl_r:.2f} | Via: {exit_via}")
        
        # Log to CSV
        try:
            import csv
            os.makedirs('analysis', exist_ok=True)
            metrics_file = 'analysis/trade_metrics.csv'
            file_exists = os.path.exists(metrics_file)
            
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'trade_id', 'symbol', 'direction', 'entry_price', 'exit_price', 'pnl_r', 'atr', 'exit_via', 'exchange_time'])
                writer.writerow([
                    datetime.now().isoformat(),
                    trade_id,
                    symbol,
                    direction,
                    entry_price,
                    exit_price,
                    round(pnl_r, 2),
                    atr,
                    exit_via,
                    exit_time.isoformat() if exchange_time else ''
                ])
        except Exception as e:
            self.logger.error(f"Metrics logging failed: {e}")
        
        # Notify orchestrator
        if self.orchestrator:
            self.orchestrator.register_exit(symbol)

    # =========================================================================
    # v2.1 FIX: RATE-LIMIT EMERGENCY FALLBACK FOR TSL
    # =========================================================================
    async def _emergency_market_close(self, symbol: str, position: Dict[str, Any], reason: str) -> bool:
        """
        Emergency market close when TSL update fails.
        v2.1: Last resort to prevent naked exposure.
        """
        trade_id = position.get('trade_id', 'UNKNOWN')
        direction = position.get('direction', 'LONG')
        position_qty = position.get('position_qty', 0)
        
        close_side = "SELL" if direction == "LONG" else "BUY"
        
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            try:
                # Get current position size from exchange
                pos_info = self.client.futures_position_information(symbol=symbol)
                current_qty = abs(float(pos_info[0]['positionAmt'])) if pos_info else position_qty
                
                if current_qty == 0:
                    self.logger.info(f"EMERGENCY [{trade_id}]: Position already closed")
                    return True
                
                # Market close
                result = self.client.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    type="MARKET",
                    quantity=current_qty,
                    reduceOnly=True
                )
                
                exit_price = float(result.get('avgPrice', 0))
                self.logger.critical(f"🚨 EMERGENCY_EXIT [{trade_id}] {symbol}: MARKET @ {exit_price} | Reason: {reason}")
                
                # v2.1: Mark as recently exited
                self._recently_exited[symbol] = time.time()
                
                # Telegram alert
                if self.telegram:
                    try:
                        await self.telegram.send_status_update(
                            f"🚨 <b>EMERGENCY EXIT</b>\n"
                            f"Symbol: {symbol}\n"
                            f"Reason: {reason}\n"
                            f"Price: {exit_price}"
                        )
                    except:
                        pass
                
                return True
                
            except BinanceAPIException as e:
                if e.code == -1015:  # Rate limit
                    self.logger.warning(f"EMERGENCY [{trade_id}]: Rate limited, retry {attempt+1}/{RATE_LIMIT_MAX_RETRIES}")
                    await asyncio.sleep(RATE_LIMIT_RETRY_MS / 1000)
                else:
                    self.logger.error(f"EMERGENCY [{trade_id}]: API error {e.code}: {e.message}")
                    break
            except Exception as e:
                self.logger.error(f"EMERGENCY [{trade_id}]: Failed: {e}")
                break
        
        # All retries failed
        self.logger.critical(f"🔴 CRITICAL_UNPROTECTED_POSITION [{trade_id}] {symbol}: Emergency close FAILED")
        if self.telegram:
            try:
                await self.telegram.send_status_update(
                    f"🔴 <b>CRITICAL: UNPROTECTED POSITION</b>\n"
                    f"Symbol: {symbol}\n"
                    f"Emergency close failed. Manual intervention required!"
                )
            except:
                pass
        return False

    # =========================================================================
    # ATOMIC TSL CHECK
    # =========================================================================
    def _is_order_still_open(self, symbol: str, order_id: int) -> bool:
        """Check if order is still open before attempting to cancel."""
        try:
            order_info = self.client.futures_get_order(symbol=symbol, orderId=order_id)
            return order_info['status'] in ['NEW', 'PARTIALLY_FILLED']
        except Exception as e:
            self.logger.debug(f"Order status check failed for {order_id}: {e}")
            return False

    # =========================================================================
    # ENTRY HANDLING
    # =========================================================================
    async def execute_order(self, signal: Dict[str, Any]) -> None:
        """Execute a trade entry with duplicate protection."""
        symbol = signal.get('symbol')
        direction = signal.get('direction')
        entry_price = signal.get('entry_price')
        atr = signal.get('atr', 1)
        size_multiplier = signal.get('size_multiplier', 1)
        timestamp = signal.get('timestamp')
        bar_range = signal.get('bar_range', 0)
        
        # Kill switch check
        if should_block_execution(self.state):
            self.logger.warning("⛔ Execution blocked: Kill switch active")
            return
        
        # v2.0: Entry lock to prevent duplicate entries
        async with self._entry_lock:
            if symbol in self.state.active_orders:
                self.logger.warning(f"⛔ Duplicate entry blocked for {symbol}")
                return
            
            trade_id = generate_trade_id()
            t_received = time.time()
            t_signal = parse_signal_timestamp(timestamp)
            latency_int_ms = compute_latency_ms(t_received, t_signal)
            
            if is_latency_exceeded(latency_int_ms):
                self.logger.warning(f"⚠️ [{trade_id}] Signal too stale ({latency_int_ms:.0f}ms)")
                return
            
            order_side = "BUY" if direction == "LONG" else "SELL"
            quantity = compute_quantity(BASE_SIZE_USD, size_multiplier, entry_price)
            
            try:
                t_sent = time.time()
                result = self.client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type="MARKET",
                    quantity=quantity
                )
                t_ack = time.time()
                
                fill_price = float(result.get('avgPrice', entry_price))
                filled_qty = float(result.get('executedQty', quantity))
                latency_net_ms = compute_latency_ms(t_ack, t_sent)
                slippage_bp = compute_slippage_bp(fill_price, entry_price)
                
                # Calculate initial stop
                initial_stop = fill_price - (atr * 1.5) if direction == 'LONG' else fill_price + (atr * 1.5)
                hard_stop = compute_hard_stop_price(fill_price, direction)
                
                # Place hard stop order
                stop_side = "SELL" if direction == "LONG" else "BUY"
                hard_stop_result = self.client.futures_create_order(
                    symbol=symbol,
                    side=stop_side,
                    type="STOP_MARKET",
                    stopPrice=round(hard_stop, 2),
                    quantity=filled_qty,
                    closePosition=False
                )
                
                # Place trailing stop order
                sl_stop_price = round(initial_stop, 2)
                sl_limit_price = round(sl_stop_price * 0.995, 2) if direction == 'LONG' else round(sl_stop_price * 1.005, 2)
                sl_result = self.client.futures_create_order(
                    symbol=symbol,
                    side=stop_side,
                    type="STOP",
                    stopPrice=sl_stop_price,
                    price=sl_limit_price,
                    quantity=filled_qty,
                    timeInForce="GTC"
                )
                
                # Store position state
                self.state.active_orders[symbol] = {
                    'trade_id': trade_id,
                    'direction': direction,
                    'entry_price': fill_price,
                    'position_qty': filled_qty,  # v2.1: Store position size for partial fill tracking
                    'stop_price': initial_stop,
                    'hard_stop': hard_stop,
                    'atr': atr,
                    'sl_order_id': sl_result.get('orderId') or sl_result.get('algoId'),
                    'hard_stop_id': hard_stop_result.get('orderId') or hard_stop_result.get('algoId'),
                    'entry_time': time.time()  # v2.1: Use local time for entry, exchange time for exits
                }
                save_state_to_file(self.state_file, self.state.active_orders)
                
                self.logger.info(f"ENTRY [{trade_id}]: {symbol} {direction} @ {fill_price:.2f} | Qty: {filled_qty} | Latency: {latency_net_ms:.0f}ms")
                
                # Telegram notification
                if self.telegram:
                    try:
                        await self.telegram.send_entry_alert(
                            pair=symbol,
                            direction=direction,
                            entry_price=fill_price,
                            stop_price=initial_stop,
                            timestamp=datetime.now()
                        )
                    except Exception as e:
                        self.logger.warning(f"Entry alert failed: {e}")
                
            except BinanceAPIException as e:
                self.logger.error(f"Entry failed [{trade_id}] {symbol}: {e.code} - {e.message}")
            except Exception as e:
                self.logger.error(f"Entry failed [{trade_id}] {symbol}: {e}")

    def trigger_kill_switch(self) -> None:
        """Trigger the kill switch."""
        self.state.kill_switch_active = True
        write_kill_flag(self.kill_flag_file)
        self.logger.critical("🛑 KILL SWITCH ACTIVATED")

    # =========================================================================
    # TRAILING STOP UPDATE LOOP
    # =========================================================================
    async def update_trailing_stops(self) -> None:
        """Main loop for trailing stop updates with rate-limit protection."""
        while True:
            await asyncio.sleep(TRAILING_LOOP_INTERVAL_SEC)
            if should_block_execution(self.state):
                continue
            
            for symbol in list(self.state.active_orders.keys()):
                if symbol not in self.state.active_orders:
                    continue
                
                position = self.state.active_orders[symbol]
                trade_id = position.get('trade_id', 'UNKNOWN')
                
                # v2.1: Skip recon-discovered positions without stop orders
                # These positions were found on exchange but we don't have stop orders for them
                sl_order_id = position.get('sl_order_id')
                hard_stop_id = position.get('hard_stop_id')
                
                if not sl_order_id and not hard_stop_id:
                    # Recon-discovered position - log and skip (no stop orders to manage)
                    if trade_id.startswith('RECON_'):
                        self.logger.debug(f"Skipping recon-discovered position {symbol} (no stop orders)")
                    continue
                exit_price = None
                exit_via = None
                
                for order_key, order_id in [('sl_order_id', sl_order_id), ('hard_stop_id', hard_stop_id)]:
                    if order_id:
                        try:
                            order_info = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                            if order_info['status'] == 'FILLED':
                                exit_price = float(order_info['avgPrice'])
                                exit_via = 'ORDER_POLL'
                                exchange_time = int(order_info.get('updateTime', 0))
                                self.logger.info(f"📊 [{trade_id}] Order {order_id} FILLED @ {exit_price} (detected via poll)")
                                break
                        except Exception as e:
                            self.logger.debug(f"Order status check failed for {order_id}: {e}")
                
                # If exit detected via polling
                if exit_price is not None:
                    with self._exit_lock:
                        if symbol not in self.state.active_orders:
                            continue  # Already handled by userDataStream
                        
                        self._recently_exited[symbol] = time.time()
                        del self.state.active_orders[symbol]
                        save_state_to_file(self.state_file, self.state.active_orders)
                    
                    await self._handle_exit(
                        symbol=symbol,
                        position=position,
                        exit_price=exit_price,
                        exit_via=exit_via,
                        exchange_time=exchange_time
                    )
                    continue
                
                # Position still open - check for trailing stop update
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                except Exception as e:
                    self.logger.error(f"Ticker fetch failed [{trade_id}] {symbol}: {e}")
                    continue
                
                # v2.1: Use .get() with defaults to handle recon positions safely
                current_stop_price = position.get('stop_price')
                current_atr = position.get('atr')
                
                # Skip if missing critical fields for TSL calculation
                if current_stop_price is None or current_atr is None:
                    self.logger.debug(f"Skipping TSL for {symbol} (missing stop_price or atr)")
                    continue
                
                new_stop, should_update = compute_trailing_stop(
                    current_price, 
                    current_stop_price, 
                    current_atr, 
                    position['direction']
                )
                
                if not should_update:
                    continue
                
                # ATOMIC TSL: Verify order still open before cancel
                if not self._is_order_still_open(symbol, sl_order_id):
                    self.logger.warning(f"⚠️ [{trade_id}] SL order {sl_order_id} already filled/canceled")
                    continue
                
                # v2.1: TSL update with rate-limit protection
                try:
                    # Cancel old stop order
                    self.client.futures_cancel_order(symbol=symbol, orderId=sl_order_id)
                    
                    # Place new stop order
                    stop_side = "SELL" if position['direction'] == 'LONG' else "BUY"
                    stop_price = round(new_stop, 2)
                    stop_limit = round(stop_price * 0.995, 2) if position['direction'] == 'LONG' else round(stop_price * 1.005, 2)
                    
                    pos_info = self.client.futures_position_information(symbol=symbol)
                    quantity = abs(float(pos_info[0]['positionAmt'])) if pos_info else 1
                    
                    try:
                        res = self.client.futures_create_order(
                            symbol=symbol,
                            side=stop_side,
                            type="STOP",
                            stopPrice=stop_price,
                            price=stop_limit,
                            quantity=quantity,
                            timeInForce="GTC"
                        )
                        
                        # Update state with new stop
                        self.state.active_orders[symbol]['stop_price'] = new_stop
                        self.state.active_orders[symbol]['sl_order_id'] = res.get('orderId') or res.get('algoId')
                        save_state_to_file(self.state_file, self.state.active_orders)
                        self.logger.info(f"Stepped Stop [{trade_id}] {symbol}: {new_stop:.2f}")
                        
                    except BinanceAPIException as e:
                        if e.code == -1015:  # Rate limit
                            # v2.1: EMERGENCY FALLBACK - position is naked, close immediately
                            self.logger.error(f"🚨 [{trade_id}] TSL replace rate-limited, initiating EMERGENCY CLOSE")
                            await self._emergency_market_close(symbol, position, "RATE_LIMIT_TSL_REPLACE")
                            
                            # Remove from tracking (emergency close handles notification)
                            with self._exit_lock:
                                if symbol in self.state.active_orders:
                                    del self.state.active_orders[symbol]
                                    save_state_to_file(self.state_file, self.state.active_orders)
                        else:
                            self.logger.error(f"TSL replace failed [{trade_id}]: {e.code} - {e.message}")
                            # Position still has hard stop as backup
                            
                except BinanceAPIException as e:
                    self.logger.error(f"TSL cancel failed [{trade_id}]: {e.code} - {e.message}")
                except Exception as e:
                    self.logger.error(f"Stop Update Fail [{trade_id}]: {e}")
