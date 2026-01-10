# main.py
"""
SOL-PERP v0.5.1 Main Orchestrator
Entry point for paper trading with decay management
"""

import asyncio
import logging
import signal
import sys
import os
import json
from datetime import datetime

from websocket_handler_improved import BinanceWebSocketFeed
from live_event_detector import LiveEventDetector
from telegram_bot import TelegramBot
from trade_executor_paper import PaperTradeExecutor
from strategy_lifecycle_manager import StrategyLifecycleManager, run_lifecycle_scheduler
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class PaperTradingSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        # Validate config
        self._validate_config()
        
        # Initialize components
        self.telegram = TelegramBot()
        self.event_detector = None
        self.feed_handler = None
        self.trade_executor = None
        self.lifecycle_manager = None
        
        # State
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("System initialized")
    
    def _validate_config(self):
        """Check for valid configuration"""
        if not TELEGRAM_TOKEN or "YOUR_BOT_TOKEN" in TELEGRAM_TOKEN:
            print("❌ ERROR: config.py missing valid TELEGRAM_TOKEN")
            sys.exit(1)
        if not TELEGRAM_CHAT_ID or "YOUR_CHAT_ID" in TELEGRAM_CHAT_ID:
            print("❌ ERROR: config.py missing valid TELEGRAM_CHAT_ID")
            sys.exit(1)
        
        # Create baseline params if not exists
        if not os.path.exists('config/trade_params_baseline.json'):
            os.makedirs('config', exist_ok=True)
            baseline = {
                "cooldown_minutes": 15,
                "max_trades_per_day": 5,
                "hold_times": [5, 15, 30, 60],
                "transaction_cost": 0.000075,
                "stop_loss_atr_mult": 1.5,
                "take_profit_atr_mult": 4.5,
                "min_atr": 0.15,
                "max_atr": 2.0,
                "trend_filter": True,
                "volume_confirmation": 1.0,
                "session_filter": [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
            }
            with open('config/trade_params_baseline.json', 'w') as f:
                json.dump(baseline, f, indent=2)
    
    def _setup_logging(self):
        """Configure system-wide logging"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(log_dir, f"system_{datetime.utcnow().strftime('%Y%m%d')}.log"),
                    mode='a'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start all components"""
        self.logger.info("=" * 80)
        self.logger.info("SOL-PERP v0.5.1 PAPER TRADING SYSTEM STARTING")
        self.logger.info("=" * 80)
        
        self.running = True
        
        # 1. Create trade executor with 2026 params
        self.trade_executor = PaperTradeExecutor(
            telegram_bot=self.telegram,
            params_path='config/trade_params_2026_enhanced.json'
        )
        
        # 2. Create event detector with integrated telegram
        self.event_detector = LiveEventDetector(
            event_callback=self.trade_executor.on_event,
            enable_telegram=True,
            telegram_bot=self.telegram
        )
        
        # 3. Create feed handler and connect event detector
        self.feed_handler = BinanceWebSocketFeed(symbol='SOLUSDT')
        
        # Monkey-patch the resample method to feed events
        original_resample = self.feed_handler._resample_to_5min
        
        def enhanced_resample():
            result = original_resample()
            
            # Send new bar to event detector
            if self.feed_handler.five_min_buffer:
                latest_bar = self.feed_handler.five_min_buffer[-1]
                self.event_detector.on_5min_bar(latest_bar)
            
            return result
        
        self.feed_handler._resample_to_5min = enhanced_resample
        
        # 4. Create lifecycle manager and start scheduler
        self.lifecycle_manager = StrategyLifecycleManager(
            trade_log_path=self.trade_executor.trade_log_path,
            telegram_bot=self.telegram
        )
        
        # Start scheduler in background
        asyncio.create_task(run_lifecycle_scheduler(self.lifecycle_manager))
        
        # 5. Start feed handler
        self.feed_handler.start()
        
        self.logger.info("✓ All components started")
        self.logger.info("Press Ctrl+C to shutdown gracefully")
        
        # Send startup notification
        await self._send_startup_notification()
        
        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Graceful shutdown"""
        if not self.running:
            return
        
        self.logger.info("Shutting down system...")
        self.running = False
        
        if self.feed_handler:
            self.feed_handler.stop()
        
        if self.trade_executor:
            await self.trade_executor.finalize()
        
        self.logger.info("✓ System stopped")
    
    def signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM"""
        self.logger.info("Received interrupt signal")
        asyncio.create_task(self.stop())
        sys.exit(0)
    
    async def _send_startup_notification(self):
        """Send Telegram notification on successful startup"""
        try:
            # Get current status
            status = self.feed_handler.get_status() if self.feed_handler else {'connected': False}
            
            message = (
                f"🚀 <b>SOL-PERP v0.5.1 System Online</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Status:</b> {'Connected' if status['connected'] else 'Starting'}\n"
                f"🕐 <b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                f"📈 <b>Symbol:</b> SOLUSDT\n"
                f"⚡ <b>Decay Radar:</b> Active\n"
                f"🤖 <b>Auto-Corrections:</b> Enabled\n"
                f"📋 <b>Trade Log:</b> {self.trade_executor.trade_log_path if self.trade_executor else 'N/A'}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ System ready for live market data"
            )
            
            await self.telegram._send_message(message)
            self.logger.info("Startup notification sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send startup notification: {e}")

# Main execution
if __name__ == "__main__":
    system = PaperTradingSystem()
    signal.signal(signal.SIGINT, system.signal_handler)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        print("\nShutdown complete")