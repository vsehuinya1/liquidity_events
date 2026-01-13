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
from strategy_orchestrator import StrategyOrchestrator
from telegram_bot import TelegramBot
# from live_event_detector import LiveEventDetector # DEPRECATED
# from trade_executor_paper import PaperTradeExecutor # DEPRECATED
# from strategy_lifecycle_manager import StrategyLifecycleManager # DEPRECATED
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
        
        # 1. Define Active Universe
        self.active_pairs = ['SOLUSDT', 'ETHUSDT', 'AVAXUSDT', 'DOGEUSDT', 'SUIUSDT'] 
        self.max_trades = 3

        # 2. Create Strategy Orchestrator (manages the fleet)
        self.orchestrator = StrategyOrchestrator(
            symbols=self.active_pairs,
            max_active_trades=self.max_trades,
            enable_telegram=True
        )
        self.telegram = self.orchestrator.telegram # Share the bot instance

        # 3. Create Multi-Pair Feed Handler
        self.feed_handler = BinanceWebSocketFeed(symbols=self.active_pairs)
        
        # Monkey-patch the resample callback to feed the orchestrator
        # The orchestrator expects on_bar_update(bar)
        self.feed_handler.on_5min_bar_callback = self.orchestrator.on_bar_update
        
        # 4. Start feed handler
        self.feed_handler.start()
        
        self.logger.info("✓ All components started")
        self.logger.info(f"✓ Active Universe: {self.active_pairs}")
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
                f"🚀 <b>GEM FLEET v1.0.0 Online</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Status:</b> {'Connected' if status['connected'] else 'Starting'}\n"
                f"🕐 <b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                f"🌎 <b>Universe:</b> {', '.join(self.active_pairs)}\n"
                f"🛡️ <b>Risk Guard:</b> Max {self.max_trades} Active Trades\n"
                f"💎 <b>Strategy:</b> Sweep + Cluster + P.Extreme\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ System ready for multi-pair scanning"
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