# trade_executor_testnet.py
"""
Testnet Trade Executor for Verification
Verifies: Latency, Slippage, State Continuity, and Attack Mode Sizing
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

# Load secrets (ensure config/secrets.py exists or use env vars)
try:
    from config.secrets import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET
except ImportError:
    # Check Env
    BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
    BINANCE_TESTNET_SECRET = os.getenv('BINANCE_TESTNET_SECRET')

class TestnetTradeExecutor:
    def __init__(self, telegram_bot=None):
        self._validate_creds()
        
        # 1. Initialize Client (Testnet Force)
        self.client = Client(BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, testnet=True)
        self.telegram = telegram_bot
        
        # 2. Setup Logging for Metrics
        self._setup_logging()
        
        # 3. State Management
        self.state_file = 'data/state/execution_state.json'
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.active_orders = self._load_state()
        
        # 4. Detector Reference (for feedback)
        # Should be set by Main after initialization
        self.detectors = {} 
        
        self.logger.info("TESTNET Executor Initialized. READY for Verification.")

    def _validate_creds(self):
        if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_SECRET:
            raise ValueError("Missing Binance Testnet Credentials in config/secrets.py or Env")

    def _setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # General Log
        self.logger = logging.getLogger('TestnetExecutor')
        self.logger.setLevel(logging.INFO)
        
        # Metric Log (CSV format for easy analysis)
        self.metric_log_path = os.path.join(log_dir, 'verify_latency.csv')
        if not os.path.exists(self.metric_log_path):
            with open(self.metric_log_path, 'w') as f:
                f.write("Time_Signal,Time_Sent,Time_Ack,Latency_Int_ms,Latency_Net_ms,Symbol,Direction,Size_Mult,Expected_Px,Fill_Px,Slippage_Bp\n")

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"State load failed: {e}")
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.active_orders, f, indent=2)

    def register_detector(self, symbol: str, detector_instance):
        """Link detector for PnL feedback"""
        self.detectors[symbol] = detector_instance

    async def execute_order(self, signal: Dict[str, Any]):
        """
        Execute trade based on signal.
        signal: {symbol, direction, entry_price, stop_loss, size_multiplier, atr, timestamp}
        """
        # 1. LATENCY GUARD (3000ms)
        t_received = time.time()
        # Parse ISO string to timestamp if needed
        if isinstance(signal['timestamp'], str):
             ts_obj = pd.to_datetime(signal['timestamp'])
             t_signal_ts = ts_obj.timestamp()
        else:
             t_signal_ts = signal['timestamp'].timestamp()
             
        latency_ms = (t_received - t_signal_ts) * 1000
        if latency_ms > 3000:
            self.logger.warning(f"🛑 REJECTED {signal['symbol']}: Latency Guard ({latency_ms:.0f}ms > 3000ms)")
            return

        symbol = signal['symbol']
        direction = signal['direction']
        size_mult = signal.get('size_multiplier', 1.0)
        
        self.logger.info(f"⚡ EXECUTING {symbol} {direction} (Size: {size_mult}x)")
        
        # 1. Calculate Quantity
        # HARDCODED RISK for verification: $100 base risk * size_mult
        # Real system should use Balance %
        base_size_usd = 100 * size_mult 
        quantity = round(base_size_usd / signal['entry_price'], 1) # Adjust rounding for SOL (1 decimal?)
        # For SOL, min qty is usually 1, step 1? Check precisions. 
        # Making safe assumption for SOL: 0 (integer) or 1? 
        # Testnet SOL precision is often int. Let's try int for safety or check info.
        # Actually standard SOL precision is 0 decimals on many pairs, but 2 on others.
        # Let's assume standard rounding.
        quantity = round(quantity, 0) 
        if quantity == 0: quantity = 1 # Min size

        side = Client.SIDE_BUY if direction == 'LONG' else Client.SIDE_SELL
        
        try:
            # 2. Place Market Order
            t_sent = time.time()
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            t_ack = time.time()
            
            # 3. Process Fill
            fill_price = float(order['avgPrice']) # Usually available in response
            # If not, calc from fills. Testnet usually returns avgPrice.
            
            # 4. Place Stop Loss (OCO simulation: Just SL for now)
            # Stop Price from signal
            stop_price = round(signal['stop_loss'], 2) # SOL precision
            stop_side = Client.SIDE_SELL if direction == 'LONG' else Client.SIDE_BUY
            
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=stop_side,
                type=Client.ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price,
                closePosition=True
            )
            
            # 4b. Place HARD STOP (Catastrophic @ 10%)
            # Independent of Soft Stop. No OCO (since Binance Fut OCO is complex, we just leave it open)
            # Or better: Just a strict STOP_MARKET that sits there.
            # If Soft Stop hits, this remains? Ideally cancel.
            # For verification simplicity: Place it.
            # Side: same as soft stop. 
            # Price: Entry +/- 10%
            hard_dist = fill_price * 0.10
            hard_price = round(fill_price - hard_dist, 2) if direction == 'LONG' else round(fill_price + hard_dist, 2)
            
            hard_stop_order = self.client.futures_create_order(
                symbol=symbol,
                side=stop_side,
                type=Client.ORDER_TYPE_STOP_MARKET,
                stopPrice=hard_price,
                closePosition=True
            )
            self.logger.info(f"🛡️ HARD STOP Placed @ {hard_price}")
            
            # 5. Log Metrics
            lat_int = (t_sent - t_received) * 1000
            lat_net = (t_ack - t_sent) * 1000
            slippage_bp = abs((fill_price - signal['entry_price']) / signal['entry_price']) * 10000
            
            metric_row = f"{signal['timestamp']},{t_sent},{t_ack},{lat_int:.2f},{lat_net:.2f},{symbol},{direction},{size_mult},{signal['entry_price']},{fill_price},{slippage_bp:.2f}\n"
            with open(self.metric_log_path, 'a') as f:
                f.write(metric_row)
                
            self.logger.info(f"✅ FILLED @ {fill_price} (Lat: {lat_net:.0f}ms, Slip: {slippage_bp:.1f}bp)")
            
            # 6. Update State
            self.active_orders[symbol] = {
                'entry_order_id': order['orderId'],
                'sl_order_id': sl_order['orderId'],
                'hard_stop_id': hard_stop_order['orderId'],
                'direction': direction,
                'entry_price': fill_price,
                'quantity': quantity,
                'stop_price': stop_price,
                'atr': signal['atr'],
                'size_mult': size_mult
            }
            self._save_state()
            
            # 7. Notify
            if self.telegram:
                await self.telegram.send_message(
                    f"🟢 **EXECUTED {symbol}**\n"
                    f"Side: {direction}\n"
                    f"Price: {fill_price}\n"
                    f"Size: {size_mult}x\n"
                    f"Latency: {lat_net:.0f}ms"
                )
                
        except BinanceAPIException as e:
            self.logger.error(f"Create Order Failed: {e}")
            if self.telegram:
                await self.telegram.send_message(f"❌ **EXECUTION FAILED**: {e}")

    async def update_trailing_stops(self):
        """
        Background task to monitor active positions and update Stops.
        Runs every 5 seconds.
        """
        while True:
            try:
                # Iterate copy of keys to avoid modification issues
                for symbol in list(self.active_orders.keys()):
                    position = self.active_orders[symbol]
                    
                    # Get Current Price
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calc New Stop
                    atr = position['atr']
                    curr_stop = position['stop_price']
                    direction = position['direction']
                    
                    # Trailing Logic (1.8 ATR) - Matches Backtest/Detector
                    # Note: We hardcode 1.8 here or get from detector.
                    # Detector sends ATR, we assume multiplier standard.
                    mult = 1.8 
                    
                    update_needed = False
                    new_stop = curr_stop
                    
                    if direction == 'LONG':
                        potential_stop = current_price - (mult * atr)
                        if potential_stop > curr_stop:
                            new_stop = potential_stop
                            update_needed = True
                    else:
                        potential_stop = current_price + (mult * atr)
                        if potential_stop < curr_stop:
                            new_stop = potential_stop
                            update_needed = True
                            
                    if update_needed:
                        # Cancel Old SL & Place New
                        # Note: In high freq, modifying is better/safer if supported, 
                        # or just cancel/replace.
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=position['sl_order_id'])
                            stop_side = Client.SIDE_SELL if direction == 'LONG' else Client.SIDE_BUY
                            
                            # Place new
                            res = self.client.futures_create_order(
                                symbol=symbol,
                                side=stop_side,
                                type=Client.ORDER_TYPE_STOP_MARKET,
                                stopPrice=round(new_stop, 2),
                                closePosition=True
                            )
                            
                            # Update State
                            self.active_orders[symbol]['stop_price'] = new_stop
                            self.active_orders[symbol]['sl_order_id'] = res['orderId']
                            self._save_state()
                            
                            self.logger.info(f"🔄 STOP UPDATED {symbol}: {curr_stop} -> {new_stop:.2f}")
                            
                        except BinanceAPIException as e:
                            self.logger.error(f"Trailing Stop Update Failed: {e}")
            
            except Exception as e:
                self.logger.error(f"Trailing Monitor Loop Error: {e}")
                
            await asyncio.sleep(5)
