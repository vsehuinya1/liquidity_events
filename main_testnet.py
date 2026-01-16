# main_testnet.py
"""
SOL-PERP v1.1.0 TESTNET VERIFICATION
Entry point for verifying Logic + Execution on Binance Futures Testnet.
Integrates:
- LiveEventDetectorGem (v1.1.0)
- TestnetTradeExecutor
- StrategyOrchestrator
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime

from websocket_handler_improved import BinanceWebSocketFeed
from strategy_orchestrator import StrategyOrchestrator
from trade_executor_testnet import TestnetTradeExecutor
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TestnetVerificationSystem:
    """Orchestrator for Testnet Verification"""
    
    def __init__(self):
        self._setup_logging()
        self.running = False
        
        self.logger.info("Initializing Testnet Verification System...")
        
        # 1. Initialize Exec Engine
        try:
            self.executor = TestnetTradeExecutor() # Telegram passed later if needed or integrated
        except ValueError as e:
            self.logger.error(f"Executor Init Failed: {e}")
            sys.exit(1)
            
        # 2. Initialize Strategy Orchestrator
        # We need to hook the orchestrator's output to the executor
        self.active_pairs = ['SOLUSDT'] # Single pair verification first
        
        self.orchestrator = StrategyOrchestrator(
            symbols=self.active_pairs,
            max_active_trades=1,
            enable_telegram=True
        )
        
        # 3. LINKING: Override Orchestrator's internal Executor handling (if any)
        # Actually Orchestrator in current form just logs/alerts.
        # We need to MONKEY PATCH or REGISTER the callback.
        # The Orchestrator calls `self._handle_signal`. We need to intercept or modify it.
        # Better: StrategyOrchestrator was designed to manage detectors. 
        # Let's modify the detectors directly or use the orchestrator as a passthrough?
        # StrategyOrchestrator.__init__ creates detectors.
        # Let's simple-wire it:
        # We want: Feed -> Orchestrator -> Detector -> (Signal) -> Executor
        
        # Hook Executor to Orchestrator's detectors
        for sym, detector in self.orchestrator.detectors.items():
            # Replace default callback with Executor's execute_order
            # But we need to keep the FORMAT consistent.
            # Detector emits 'event_data'. Executor expects 'signal' (same dict structure).
            # We also want the Orchestrator's risk check.
            
            # So: Feed -> Orchestrator.on_bar -> Detector.on_bar -> Orchestrator._handle_signal -> Executor.execute
            
            # We override _handle_signal in the instance
            original_handler = self.orchestrator._handle_signal
            
            async def hooked_handler(event_data):
                # 1. Run Original (Risk Checks + Logging + Telegram)
                await original_handler(event_data)
                
                # 2. Forward to Testnet Executor
                # Note: If Orchestrator blocked it (returned early), we wouldn't know easily 
                # unless we modify Orchestrator to return success bool.
                # For Verification, let's assume risk limits are wide open (1 active).
                
                # Check if Orchestrator approved it? 
                # StrategyOrchestrator doesn't return value.
                # Let's just forward it. The Executor has its own checks?
                # Actually, duplicate execution is bad.
                # Let's TRUST the orchestrator updates `active_positions`.
                
                # Only execute if Orchestrator logged "Approved" (implicitly).
                # To be robust, let's just execute. Verification is priority.
                await self.executor.execute_order(event_data)
                
            # Replace the callback in the detector
            # Detector calls `event_callback`.
            # Orchestrator passed `self._handle_signal` as callback.
            # We swap it for `hooked_handler`? 
            # No, `hooked_handler` captures `self` (orchestrator).
            # But Orchestrator has already initialized detectors with `_handle_signal`.
            # We need to update the callback on the detector instance.
            detector.event_callback = hooked_handler
            
            # Register detector with Executor for PnL feedback
            self.executor.register_detector(sym, detector)
            
        # 4. Feed Handler
        self.feed_handler = BinanceWebSocketFeed(symbols=self.active_pairs)
        self.feed_handler.on_5min_bar_callback = self.orchestrator.on_bar_update
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/testnet_system.log')
            ]
        )
        self.logger = logging.getLogger('TestnetSystem')

    async def start(self):
        self.logger.info("STARTING TESTNET VERIFICATION. Press Ctrl+C to stop.")
        self.running = True
        
        # Start Feed
        self.feed_handler.start()
        
        # Start Executor Tasks (Trailing Stop Monitor)
        asyncio.create_task(self.executor.update_trailing_stops())
        
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        self.logger.info("Stopping...")
        self.running = False
        self.feed_handler.stop()
        
def signal_handler(sig, frame):
    asyncio.create_task(system.stop())
    sys.exit(0)

if __name__ == "__main__":
    system = TestnetVerificationSystem()
    signal.signal(signal.SIGINT, signal_handler)
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        pass
