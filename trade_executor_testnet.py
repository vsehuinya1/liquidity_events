# trade_executor_testnet.py
"""
Testnet Trade Executor for Verification
Verifies: Latency, Slippage, State Continuity, and Attack Mode Sizing
Includes: Manual /KILL Switch logic.
"""

import logging
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

# Load secrets
try:
    from config.secrets import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET
except ImportError:
    BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
    BINANCE_TESTNET_SECRET = os.getenv('BINANCE_TESTNET_SECRET')

class TestnetTradeExecutor:
    def __init__(self, telegram_bot=None):
        self._validate_creds()
        
        self.client = Client(BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, testnet=True)
        self.telegram = telegram_bot
        
        self._setup_logging()
        
        self.state_file = 'data/state/execution_state.json'
        self.kill_flag_file = 'data/state/kill_switch.flag'
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        # STATE & FLAGS
        self.active_orders = self._load_state()
        self.detectors = {} 
        
        # CHECK PERSISTENT KILL SWITCH
        self.KILL_SWITCH = False
        if os.path.exists(self.kill_flag_file):
            self.KILL_SWITCH = True
            self.logger.critical("🚨 STARTUP BLOCKED: 'kill_switch.flag' DETECTED. SYSTEM LOCKED.")
        
        if not self.KILL_SWITCH:
            self.logger.info("TESTNET Executor Initialized. READY.")
        else:
             self.logger.warning("Executor Initialized in LOCKDOWN Mode.")

    def _validate_creds(self):
        if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_SECRET:
            raise ValueError("Missing Binance Testnet Credentials")

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('TestnetExecutor')
        self.logger.setLevel(logging.INFO)
        self.metric_log_path = os.path.join(log_dir, 'verify_latency.csv')
        if not os.path.exists(self.metric_log_path):
            with open(self.metric_log_path, 'w') as f:
                f.write("Time_Signal,Time_Sent,Time_Ack,Latency_Int_ms,Latency_Net_ms,Symbol,Direction,Size_Mult,Expected_Px,Fill_Px,Slippage_Bp,Attack_Mode,Bar_Range\n")

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f: return json.load(f)
            except Exception: pass
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.active_orders, f, indent=2)

    # =========================================================================
    # KILL SWITCH LOGIC
    # =========================================================================
    async def trigger_kill_switch(self):
        """
        IRREVERSIBLE KILL SWITCH.
        1. Set Flag
        2. Cancel All Orders
        3. Force Close All Positions
        """
        self.logger.critical("🚨 EXECUTING KILL SWITCH SEQUENCE...")
        self.KILL_SWITCH = True
        
        # 1. Persist Flag
        with open(self.kill_flag_file, 'w') as f:
            f.write(f"KILLED_AT_{datetime.utcnow().isoformat()}")
            
        # 2. Cancel All Open Orders
        try:
            # Note: cancel_open_orders usually takes symbol. Loop signals?
            # Or use global cancel if available? Testnet might not support `cancel_open_orders` without symbol.
            # We will loop known active symbols + hardcoded list if needed.
            # Best effort: Loop active_orders keys.
            targets = list(self.active_orders.keys())
            if not targets: targets = ['SOLUSDT'] # Default fallback
            
            for s in targets:
                try:
                    self.client.futures_cancel_all_open_orders(symbol=s)
                    self.logger.info(f"KILLED ORDERS: {s}")
                except Exception as e:
                    self.logger.error(f"Kill Cancel Failed {s}: {e}")
                    
        except Exception as e:
             self.logger.error(f"Global Cancel Failed: {e}")
             
        # 3. Force Close Positions
        # Use account info to find ANY position, not just tracked ones
        try:
            acc = self.client.futures_account()
            for pos in acc['positions']:
                amt = float(pos['positionAmt'])
                if amt != 0:
                    sym = pos['symbol']
                    side = Client.SIDE_SELL if amt > 0 else Client.SIDE_BUY
                    try:
                        self.client.futures_create_order(
                            symbol=sym,
                            side=side,
                            type=Client.ORDER_TYPE_MARKET,
                            reduceOnly=True,
                            quantity=abs(amt)
                        )
                        self.logger.critical(f"KILLED POSITION: {sym} {amt}")
                    except Exception as e:
                        self.logger.critical(f"Kill Position Failed {sym}: {e}")
                        
        except Exception as e:
            self.logger.critical(f"Account Position Scan Failed: {e}")
            
        self.logger.critical("🚨 SYSTEM KILLED. MANUAL RESTART REQUIRED.")

    # =========================================================================
    # EXECUTION
    # =========================================================================
    async def execute_order(self, signal: Dict[str, Any]):
        # 0. KILL SWITCH GUARD
        if self.KILL_SWITCH:
            self.logger.warning("🛑 EXECUTION BLOCKED: KILL SWITCH ACTIVE")
            return

        # 1. LATENCY GUARD
        t_received = time.time()
        if isinstance(signal['timestamp'], str):
             ts_obj = pd.to_datetime(signal['timestamp'])
             t_signal_ts = ts_obj.timestamp()
        else:
             t_signal_ts = signal['timestamp'].timestamp()
             
        latency_ms = (t_received - t_signal_ts) * 1000
        if latency_ms > 3000:
            self.logger.warning(f"🛑 REJECTED {signal['symbol']}: Latency Guard ({latency_ms:.0f}ms)")
            return

        symbol = signal['symbol']
        direction = signal['direction']
        size_mult = signal.get('size_multiplier', 1.0)
        
        self.logger.info(f"⚡ EXECUTING {symbol} {direction} (Size: {size_mult}x)")
        
        base_size_usd = 100 * size_mult 
        quantity = round(base_size_usd / signal['entry_price'], 0) 
        if quantity == 0: quantity = 1
        side = Client.SIDE_BUY if direction == 'LONG' else Client.SIDE_SELL
        
        try:
            # 2. Market Entry
            t_sent = time.time()
            order = self.client.futures_create_order(
                symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, quantity=quantity
            )
            t_ack = time.time()
            fill_price = float(order.get('avgPrice', signal['entry_price']))
            
            # 3. Soft Stop
            stop_price = round(signal['stop_loss'], 2)
            stop_side = Client.SIDE_SELL if direction == 'LONG' else Client.SIDE_BUY
            sl_order = self.client.futures_create_order(
                symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price, closePosition=True
            )
            
            # 4. Hard Stop
            hard_dist = fill_price * 0.10
            hard_price = round(fill_price - hard_dist, 2) if direction == 'LONG' else round(fill_price + hard_dist, 2)
            hard_stop = self.client.futures_create_order(
                symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET,
                stopPrice=hard_price, closePosition=True
            )
            
            # 5. Log & State
            lat_net = (t_ack - t_sent) * 1000
            slippage_bp = abs((fill_price - signal['entry_price']) / signal['entry_price']) * 10000
            
            # Extract Meta
            is_attack = signal.get('meta_attack_mode', False)
            bar_range = signal.get('meta_bar_range', 0.0)
            
            metric_row = f"{signal['timestamp']},{t_sent},{t_ack},{lat_int:.2f},{lat_net:.2f},{symbol},{direction},{size_mult},{signal['entry_price']},{fill_price},{slippage_bp:.2f},{is_attack},{bar_range:.4f}\n"
            with open(self.metric_log_path, 'a') as f:
                f.write(metric_row)

            self.active_orders[symbol] = {
                'entry_order_id': order['orderId'],
                'sl_order_id': sl_order['orderId'],
                'hard_stop_id': hard_stop['orderId'],
                'direction': direction,
                'stop_price': stop_price,
                'atr': signal['atr']
            }
            self._save_state()
            
            if self.telegram:
                await self.telegram.send_entry_alert(symbol, direction, fill_price, 'entry', stop_price, 0, signal['atr'], datetime.utcnow())

        except BinanceAPIException as e:
            self.logger.error(f"Exec Failed: {e}")
            if self.telegram: await self.telegram.send_error_alert(str(e))

    async def update_trailing_stops(self):
        while True:
            # 0. KILL SWITCH GUARD
            if self.KILL_SWITCH:
                await asyncio.sleep(5)
                continue

            try:
                for symbol in list(self.active_orders.keys()):
                    position = self.active_orders[symbol]
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Logic matches previous... (abbreviated for overwrite)
                    atr = position['atr']
                    curr_stop = position['stop_price']
                    direction = position['direction']
                    mult = 1.8 
                    
                    new_stop = curr_stop
                    update = False
                    
                    if direction == 'LONG':
                        prop = current_price - (mult * atr)
                        if prop > curr_stop: new_stop, update = prop, True
                    else:
                        prop = current_price + (mult * atr)
                        if prop < curr_stop: new_stop, update = prop, True
                            
                    if update:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=position['sl_order_id'])
                            stop_side = Client.SIDE_SELL if direction == 'LONG' else Client.SIDE_BUY
                            res = self.client.futures_create_order(
                                symbol=symbol, side=stop_side, type=Client.ORDER_TYPE_STOP_MARKET,
                                stopPrice=round(new_stop, 2), closePosition=True
                            )
                            self.active_orders[symbol]['stop_price'] = new_stop
                            self.active_orders[symbol]['sl_order_id'] = res['orderId']
                            self._save_state()
                            self.logger.info(f"Step Stop {symbol}: {new_stop:.2f}")
                        except Exception as e:
                            self.logger.error(f"Stop Update Fail: {e}")
                            
            except Exception as e:
                self.logger.error(f"Trailing Loop Err: {e}")
            
            await asyncio.sleep(5)
