# main.py
"""
SOL-PERP Paper Trading System - Entry Point
Wires together: Feed Handler → Event Detector → Telegram → Paper Logger
"""

import asyncio
import logging
from datetime import datetime
import sys
import signal
import os

from websocket_handler_improved import BinanceWebSocketFeed
from live_event_detector import LiveEventDetector, integrate_with_feed_handler
from telegram_bot import TelegramBot
from trade_executor_paper import PaperTradeExecutor

# ============================================================================
# PAPER TRADING ORCHESTRATOR
# ============================================================================

class PaperTradingSystem:
    """Main orchestrator for paper trading with Telegram alerts"""
    
    def __init__(self):
        # Initialize components
        self.telegram = TelegramBot()
        self.event_detector = None  # Will be created after Telegram init
        self.feed_handler = None
        self.trade_executor = None
        
        # State management
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup system-wide logging"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, f'paper_trading_{datetime.utcnow().strftime("%Y%m%d")}.log')
                ),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the paper trading system"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING SOL-PERP PAPER TRADING SYSTEM")
        self.logger.info("=" * 60)
        
        self.running = True
        
        # Create trade executor (paper mode)
        self.trade_executor = PaperTradeExecutor(
            telegram=self.telegram,
            max_trades_per_day=5,
            daily_loss_limit_bp=-150
        )
        
        # Create event detector with integrated Telegram
        self.event_detector = LiveEventDetector(
            event_callback=self.trade_executor.on_event,
            enable_telegram=True
        )
        
        # Create feed handler with callback to event detector
        self.feed_handler = BinanceWebSocketFeed(
            symbol='SOLUSDT',
            on_5min_bar_callback=self.event_detector.on_5min_bar
        )
        
        # Start the feed handler
        self.feed_handler.start()
        
        self.logger.info("System started successfully")
        self.logger.info("Press Ctrl+C to stop gracefully")
        
        # Keep running until interrupted
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down paper trading system...")
        self.running = False
        
        if self.feed_handler:
            self.feed_handler.stop()
        
        if self.trade_executor:
            await self.trade_executor.finalize()
        
        self.logger.info("System stopped")
    
    def signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM"""
        self.logger.info("Received shutdown signal")
        asyncio.create_task(self.stop())
        sys.exit(0)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check for config
    try:
        from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("⚠️  WARNING: config.py contains placeholder token")
            print("   Please add your real Telegram token and chat ID")
            sys.exit(1)
    except ImportError:
        print("❌ ERROR: config.py not found")
        print("   Create config.py with:")
        print("   TELEGRAM_TOKEN = 'your_token'")
        print("   TELEGRAM_CHAT_ID = 'your_chat_id'")
        sys.exit(1)
    
    # Run system
    system = PaperTradingSystem()
    signal.signal(signal.SIGINT, system.signal_handler)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        print("\nShutdown complete")
