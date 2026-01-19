# main_testnet.py
"""
Testnet Verification Entry Point
Phase 2: Risk Engine + Manual Kill Switch (/KILL)
"""

import asyncio
import logging
import sys
import os

from trade_executor_testnet import TestnetTradeExecutor
from risk_manager import RiskManager
from telegram_bot import TelegramBot
from websocket_handler_improved import BinanceWebSocketFeed
from strategy_orchestrator import StrategyOrchestrator
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TestnetVerificationSystem:
    def __init__(self):
        self._setup_logging()
        self.running = False
        self.logger.info("Initializing Testnet Verification System...")
        
        # 0. Risk Engine (Phase 2)
        # Loose limits checking (-100R), Correlation guarding (1 max)
        self.risk_manager = RiskManager(daily_loss_limit_r=-100.0, max_active_correlated=1)
        
        # 1. Init Components (Bot lazy loaded in run)
        self.active_pairs = ['SOLUSDT'] 
        self.orchestrator = StrategyOrchestrator(
            symbols=self.active_pairs,
            max_active_trades=10, 
            enable_telegram=True 
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler("logs/testnet_system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("SystemMain")

    async def run(self):
        self.running = True
        
        # LIFECYCLE MANAGEMENT
        # We need the bot to be active for the Executor (notifications) AND input (polling)
        async with TelegramBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) as bot:
            
            # 2. Initialize Executor (With Bot)
            self.executor = TestnetTradeExecutor(telegram_bot=bot)
            
            # 3. WIRE KILL SWITCH
            # Connect Bot Command -> Executor Action
            bot.set_kill_callback(self.executor.trigger_kill_switch)
            
            # 4. Start Bot Polling (Background)
            poll_task = asyncio.create_task(bot.start_polling())
            
            # 5. Start Execution Loop (Trailing Stops)
            exec_task = asyncio.create_task(self.executor.update_trailing_stops())
            
            # 6. Define Signal Pipeline
            async def hooked_handler(event_data):
                # A. Log & internal checks
                # Note: We duplicate Orchestrator logic or call it? 
                # StrategyOrchestrator usually has its own handler, but we use it as state containers.
                # Actually, WS feeds data to Orchestrator, which calls 'event_callback'.
                # SO: WS -> Orchestrator._handle_bar -> (Detect) -> _handle_signal -> CALLS THIS.
                
                self.logger.info(f"Signal Received: {event_data['symbol']} {event_data['direction']}")
                
                # B. Risk Manager Gate
                validated = self.risk_manager.validate_signal(event_data)
                
                if validated:
                    # C. Execute
                    await self.executor.execute_order(validated)
                    self.risk_manager.register_entry(validated['symbol'])
                else:
                    self.logger.warning(f"⛔ Blocked by Risk Manager: {event_data['symbol']}")

            # 7. Wire Orchestrator Callback
            # The orchestrator's detectors need to call 'hooked_handler' when they see a signal.
            # StrategyOrchestrator._handle_signal calls self.event_callback if set.
            # But StrategyOrchestrator constructor doesn't take callback easily in my previous check.
            # It usually sets it on the detectors.
            # Let's iterate detectors and force set it.
            
            for symbol, detector in self.orchestrator.detectors.items():
                detector.event_callback = hooked_handler
                # Also register for PnL feedback if needed
                self.executor.register_detector(symbol, detector)

            # 8. Start WebSocket
            self.logger.info(f"Connecting to Binance Testnet Stream for {self.active_pairs}...")
            feed = BinanceWebSocketFeed(
                symbols=self.active_pairs,
                on_5min_bar_callback=self.orchestrator.on_bar_update
            )
            
            # Start WS Feed (runs as daemon thread, non-async)
            feed.start()
            
            self.logger.info("✅ SYSTEM LIVE. Listening for Signals & /KILL command...")
            
            # Keep alive (poll + trailing stops)
            try:
                await asyncio.gather(poll_task, exec_task)
            except asyncio.CancelledError:
                self.logger.info("System Shutdown Initiated")
                feed.stop()
            except Exception as e:
                self.logger.critical(f"System Crash: {e}")
                
if __name__ == "__main__":
    sys = TestnetVerificationSystem()
    try:
        asyncio.run(sys.run())
    except KeyboardInterrupt:
        print("Manual Stop")
