# strategy_orchestrator.py
"""
Strategy Orchestrator: Fleet Manager for LiveEventDetectorGem instances.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from live_event_detector_gem import LiveEventDetectorGem
import asyncio
from telegram_bot import TelegramBot


@dataclass
class OrchestratorState:
    active_positions: int = 0
    symbols: tuple = field(default_factory=tuple)
    max_active_trades: int = 3


def is_at_max_capacity(active_positions: int, max_active_trades: int) -> bool:
    return active_positions >= max_active_trades


def increment_active_positions(current: int) -> int:
    return current + 1


def decrement_active_positions(current: int) -> int:
    return max(current - 1, 0)


def is_valid_symbol(symbol: Optional[str], known_symbols: Dict[str, Any]) -> bool:
    return symbol is not None and symbol in known_symbols


def setup_orchestrator_logging() -> logging.Logger:
    logger = logging.getLogger('StrategyOrchestrator')
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger


def create_detector_fleet(
    symbols: List[str],
    signal_callback: Callable,
    telegram_bot: Optional[TelegramBot]
) -> Dict[str, LiveEventDetectorGem]:
    detectors = {}
    for symbol in symbols:
        detectors[symbol] = LiveEventDetectorGem(
            symbol=symbol,
            event_callback=signal_callback,
            enable_telegram=False,
            telegram_bot=telegram_bot
        )
    return detectors


class StrategyOrchestrator:
    def __init__(
        self,
        symbols: List[str],
        max_active_trades: int = 3,
        enable_telegram: bool = True
    ):
        self.logger = setup_orchestrator_logging()
        self.state = OrchestratorState(
            active_positions=0,
            symbols=tuple(symbols),
            max_active_trades=max_active_trades
        )
        self.telegram: Optional[TelegramBot] = TelegramBot() if enable_telegram else None
        self.detectors = create_detector_fleet(
            symbols=symbols,
            signal_callback=self._handle_signal,
            telegram_bot=self.telegram
        )
        self.logger.info(f"Orchestrator initialized for {len(symbols)} pairs: {symbols}")
        self.logger.info(f"Global Risk Limit: Max {max_active_trades} active trades")

    @property
    def symbols(self) -> tuple:
        return self.state.symbols
    
    @property
    def max_active_trades(self) -> int:
        return self.state.max_active_trades
    
    @property
    def active_positions(self) -> int:
        return self.state.active_positions
    
    @active_positions.setter
    def active_positions(self, value: int) -> None:
        self.state.active_positions = value

    def on_bar_update(self, bar_data: Dict[str, Any]) -> None:
        symbol = bar_data.get('symbol')
        if not is_valid_symbol(symbol, self.detectors):
            return
        self.detectors[symbol].on_bar(bar_data)

    async def _handle_signal(self, event_data: Dict[str, Any]) -> None:
        symbol = event_data['symbol']
        direction = event_data['direction']
        
        if is_at_max_capacity(self.state.active_positions, self.state.max_active_trades):
            await self._handle_signal_rejected(symbol, direction)
            return
        
        self.state.active_positions = increment_active_positions(self.state.active_positions)
        print(f"✅ [ORCHESTRATOR] Approved Signal: {symbol} {direction}. Total Active: {self.state.active_positions}")
        # Note: Entry notification is handled by the Executor (authoritative source)

    async def _handle_signal_rejected(self, symbol: str, direction: str) -> None:
        self.logger.warning(f"⚠️ [RISK] Signal IGNORED for {symbol}: Max active trades ({self.state.max_active_trades}) reached.")
        if self.telegram:
            await self.telegram.send_message(f"⚠️ **SIGNAL IGNORED**\n{symbol} {direction} valid but capped by Max Trades ({self.state.max_active_trades}).")

        self.state.active_positions = decrement_active_positions(self.state.active_positions)
        self.logger.info(f"Position closed: {symbol}. Total Active: {self.state.active_positions}")

    def set_async_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Inject main event loop into all detectors for thread-safe callbacks."""
        for detector in self.detectors.values():
            detector.set_async_loop(loop)
