# strategy_orchestrator.py

import logging
import asyncio
from typing import Dict, Any, List
from live_event_detector_gem import LiveEventDetectorGem
from telegram_bot import TelegramBot

class StrategyOrchestrator:
    """
    Manages a fleet of LiveEventDetectorGem instances.
    - Routes feed updates to the correct symbol detector
    - Enforces global risk limits (Max Active Trades)
    - Centralizes logging and alerting
    """

    def __init__(self, symbols: List[str], max_active_trades: int = 3, enable_telegram: bool = True):
        self.logger = logging.getLogger('StrategyOrchestrator')
        self._setup_logging()
        
        self.symbols = symbols
        self.max_active_trades = max_active_trades
        self.active_positions = 0
        
        # Initialize Telegram Bot once (shared)
        self.telegram = TelegramBot() if enable_telegram else None
        
        # Initialize Fleet
        self.detectors: Dict[str, LiveEventDetectorGem] = {}
        for sym in symbols:
            self.detectors[sym] = LiveEventDetectorGem(
                symbol=sym,
                event_callback=self._handle_signal,
                enable_telegram=False, # Orchestrator handles Telegram to enforce risk check first
                telegram_bot=self.telegram
            )
        
        self.logger.info(f"Orchestrator initialized for {len(symbols)} pairs: {symbols}")
        self.logger.info(f"Global Risk Limit: Max {max_active_trades} active trades")

    def _setup_logging(self):
        log_dir = 'logs'
        # Orchestrator uses its own log or shares; detectors use their own named logs.
        # This setup is just for the Orchestrator shell itself.
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def on_bar_update(self, bar_data: Dict[str, Any]):
        """
        Route incoming 5-min bar to the correct detector.
        Expected bar_data format: {'symbol': 'SOLUSDT', ...bar_fields...}
        """
        symbol = bar_data.get('symbol')
        if not symbol or symbol not in self.detectors:
            # self.logger.warning(f"Received update for unknown symbol: {symbol}")
            return

        # Forward to specific detector
        self.detectors[symbol].on_5min_bar(bar_data)

    async def _handle_signal(self, event_data: Dict[str, Any]):
        """
        Central logic for signal execution.
        Check global risk limits before allowing trade.
        """
        symbol = event_data['symbol']
        direction = event_data['direction']
        
        # Check Global Risk
        if self.active_positions >= self.max_active_trades:
            self.logger.warning(f"⚠️ [RISK] Signal IGNORED for {symbol}: Max active trades ({self.max_active_trades}) reached.")
            if self.telegram:
                await self.telegram.send_message(f"⚠️ **SIGNAL IGNORED**\n{symbol} {direction} valid but capped by Max Trades ({self.max_active_trades}).")
            return

        # If allowed, increment active positions (Assume execution)
        # Note: In a real system, we'd wait for trade confirmation. 
        # Here we assume entry = active position.
        # We need a way to decrement this later (e.g., on exit signal or timeout).
        # For now, let's just log it. A full LifecycleManager is needed for true state tracking.
        self.active_positions += 1
        
        print(f"✅ [ORCHESTRATOR] Approved Signal: {symbol} {direction}. Total Active: {self.active_positions}")
        
        # Send Alert (since we disabled internal detector alerts to control risk)
        if self.telegram:
            await self.telegram.send_entry_alert(
                pair=symbol,
                direction=direction,
                entry_price=event_data['entry_price'],
                event_type=event_data['event_type'],
                stop_loss=event_data['stop_loss'],
                take_profit=event_data['take_profit'],
                atr=event_data['atr'],
                timestamp=event_data['timestamp']
            )
            
        # TODO: Forward to TradeExecutor if automated
