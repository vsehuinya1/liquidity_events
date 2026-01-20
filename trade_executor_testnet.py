# trade_executor_testnet.py
"""
Testnet Trade Executor for Verification
"""

import logging
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

try:
    from config.secrets import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET
except ImportError:
    BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
    BINANCE_TESTNET_SECRET = os.getenv('BINANCE_TESTNET_SECRET')

STATE_FILE_PATH = 'data/state/execution_state.json'
KILL_FLAG_FILE_PATH = 'data/state/kill_switch.flag'
METRIC_LOG_PATH = 'logs/verify_latency.csv'
METRIC_CSV_HEADER = "Time_Signal,Time_Sent,Time_Ack,Latency_Int_ms,Latency_Net_ms,Symbol,Direction,Size_Mult,Expected_Px,Fill_Px,Slippage_Bp,Attack_Mode,Bar_Range\n"
BASE_SIZE_USD = 100
LATENCY_THRESHOLD_MS = 3000
TRAILING_STOP_MULTIPLIER = 1.8
HARD_STOP_DISTANCE_PERCENT = 0.10
TRAILING_LOOP_INTERVAL_SEC = 5


@dataclass
class ExecutorState:
    kill_switch_active: bool = False
    active_orders: Dict[str, Dict[str, Any]] = field(default_factory=dict)


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


class TestnetTradeExecutor:
    def __init__(self, telegram_bot=None):
        validate_credentials()
        self.client = Client(BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, testnet=True)
        self.telegram = telegram_bot
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
        if self.state.kill_switch_active:
            self.logger.critical("🚨 STARTUP BLOCKED: 'kill_switch.flag' DETECTED. SYSTEM LOCKED.")
            self.logger.warning("Executor Initialized in LOCKDOWN Mode.")
        else:
            self.logger.info("TESTNET Executor Initialized. READY.")

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

    async def trigger_kill_switch(self) -> None:
        self.logger.critical("🚨 EXECUTING KILL SWITCH SEQUENCE...")
        self.state.kill_switch_active = True
        write_kill_flag(self.kill_flag_file)
        await self._cancel_all_orders()
        await self._close_all_positions()
        self.logger.critical("🚨 SYSTEM KILLED. MANUAL RESTART REQUIRED.")

    async def _cancel_all_orders(self) -> None:
        targets = list(self.state.active_orders.keys())
        if not targets:
            targets = ['SOLUSDT']
        for symbol in targets:
            try:
                self.client.futures_cancel_all_open_orders(symbol=symbol)
                self.logger.info(f"KILLED ORDERS: {symbol}")
            except Exception as e:
                self.logger.error(f"Kill Cancel Failed {symbol}: {e}")

    async def _close_all_positions(self) -> None:
        try:
            acc = self.client.futures_account()
            for pos in acc['positions']:
                amt = float(pos['positionAmt'])
                if amt == 0:
                    continue
                symbol = pos['symbol']
                side = Client.SIDE_SELL if amt > 0 else Client.SIDE_BUY
                try:
                    self.client.futures_create_order(symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, reduceOnly=True, quantity=abs(amt))
                    self.logger.critical(f"KILLED POSITION: {symbol} {amt}")
                except Exception as e:
                    self.logger.critical(f"Kill Position Failed {symbol}: {e}")
        except Exception as e:
            self.logger.critical(f"Account Position Scan Failed: {e}")

    async def execute_order(self, signal: Dict[str, Any]) -> None:
        if should_block_execution(self.state):
            self.logger.warning("🛑 EXECUTION BLOCKED: KILL SWITCH ACTIVE")
            return
        t_received = time.time()
        t_signal = parse_signal_timestamp(signal['timestamp'])
        latency_int_ms = compute_latency_ms(t_received, t_signal)
        if is_latency_exceeded(latency_int_ms):
            self.logger.warning(f"🛑 REJECTED {signal['symbol']}: Latency Guard ({latency_int_ms:.0f}ms)")
            return
        symbol = signal['symbol']
        direction = signal['direction']
        size_mult = signal.get('size_multiplier', 1.0)
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        atr = signal['atr']
        self.logger.info(f"⚡ EXECUTING {symbol} {direction} (Size: {size_mult}x)")
        quantity = compute_quantity(BASE_SIZE_USD, size_mult, entry_price)
        side = Client.SIDE_BUY if direction == 'LONG' else Client.SIDE_SELL
        stop_side = Client.SIDE_SELL if direction == 'LONG' else Client.SIDE_BUY
        try:
            t_sent = time.time()
            order = self.client.futures_create_order(symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, quantity=quantity)
            t_ack = time.time()
            fill_price = float(order.get('avgPrice', entry_price))
            stop_price = round(stop_loss, 2)
            sl_order = self.client.futures_create_order(symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET, stopPrice=stop_price, closePosition=True)
            hard_price = compute_hard_stop_price(fill_price, direction)
            hard_stop = self.client.futures_create_order(symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET, stopPrice=hard_price, closePosition=True)
            latency_net_ms = (t_ack - t_sent) * 1000
            slippage_bp = compute_slippage_bp(fill_price, entry_price)
            is_attack = signal.get('meta_attack_mode', False)
            bar_range = signal.get('meta_bar_range', 0.0)
            metric_row = format_metric_row(signal['timestamp'], t_sent, t_ack, latency_int_ms, latency_net_ms, symbol, direction, size_mult, entry_price, fill_price, slippage_bp, is_attack, bar_range)
            append_metric_log(self.metric_log_path, metric_row)
            self.state.active_orders[symbol] = {'entry_order_id': order['orderId'], 'sl_order_id': sl_order['orderId'], 'hard_stop_id': hard_stop['orderId'], 'direction': direction, 'stop_price': stop_price, 'atr': atr}
            save_state_to_file(self.state_file, self.state.active_orders)
            if self.telegram:
                await self.telegram.send_entry_alert(symbol, direction, fill_price, 'entry', stop_price, 0, atr, datetime.utcnow())
        except BinanceAPIException as e:
            self.logger.error(f"Exec Failed: {e}")
            if self.telegram:
                await self.telegram.send_error_alert(str(e))

    async def update_trailing_stops(self) -> None:
        while True:
            if should_block_execution(self.state):
                await asyncio.sleep(TRAILING_LOOP_INTERVAL_SEC)
                continue
            try:
                await self._process_trailing_stops()
            except Exception as e:
                self.logger.error(f"Trailing Loop Err: {e}")
            await asyncio.sleep(TRAILING_LOOP_INTERVAL_SEC)

    async def _process_trailing_stops(self) -> None:
        for symbol in list(self.state.active_orders.keys()):
            position = self.state.active_orders[symbol]
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            new_stop, should_update = compute_trailing_stop(current_price, position['stop_price'], position['atr'], position['direction'])
            if not should_update:
                continue
            try:
                self.client.futures_cancel_order(symbol=symbol, orderId=position['sl_order_id'])
                stop_side = Client.SIDE_SELL if position['direction'] == 'LONG' else Client.SIDE_BUY
                res = self.client.futures_create_order(symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET, stopPrice=round(new_stop, 2), closePosition=True)
                self.state.active_orders[symbol]['stop_price'] = new_stop
                self.state.active_orders[symbol]['sl_order_id'] = res['orderId']
                save_state_to_file(self.state_file, self.state.active_orders)
                self.logger.info(f"Step Stop {symbol}: {new_stop:.2f}")
            except Exception as e:
                self.logger.error(f"Stop Update Fail: {e}")
