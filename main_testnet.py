# main_testnet.py
"""
Testnet Verification Entry Point
Phase 2: Risk Engine + Manual Kill Switch (/KILL)

Architecture:
    - SystemState: Explicit, serializable system state container
    - Signal Pipeline: signal detection -> risk validation -> execution -> telemetry
    - Kill Switch: Injected callback, impossible to bypass once triggered
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any

from trade_executor_testnet import TestnetTradeExecutor
from risk_manager import RiskManager
from telegram_bot import TelegramBot
from websocket_handler_improved import BinanceWebSocketFeed
from strategy_orchestrator import StrategyOrchestrator
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


# =============================================================================
# STATE CONTAINER
# =============================================================================

@dataclass
class SystemState:
    """
    Explicit, serializable system state.
    All fields are deterministic and restart-safe.
    """
    running: bool = False
    kill_switch_triggered: bool = False
    active_pairs: tuple = field(default_factory=lambda: ('SOLUSDT',))
    # Immutable config
    daily_loss_limit_r: float = -10.0
    max_active_correlated: int = 1
    max_active_trades: int = 10


# =============================================================================
# PURE FUNCTIONS: SIGNAL PIPELINE
# =============================================================================

def validate_signal_with_risk_manager(
    risk_manager: RiskManager,
    event_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Pure risk validation gate.
    Returns validated signal dict or None if blocked.
    No side effects.
    """
    return risk_manager.validate_signal(event_data)


def should_block_execution(state: SystemState) -> bool:
    """
    Explicit kill-switch check.
    Must be called before any execution path.
    """
    return state.kill_switch_triggered or not state.running


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure and return the system logger.
    Side effect: Creates log file, configures root logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler("logs/testnet_system.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("SystemMain")


# =============================================================================
# SIGNAL HANDLER FACTORY
# =============================================================================

def create_signal_handler(
    state: SystemState,
    risk_manager: RiskManager,
    executor: TestnetTradeExecutor,
    logger: logging.Logger
) -> Callable:
    """
    Factory: Returns async signal handler with explicit dependencies.
    
    Pipeline:
        1. Kill-switch gate (cannot bypass)
        2. Telemetry (log signal receipt)
        3. Risk validation
        4. Execution
        5. State registration
    """
    async def handle_signal(event_data: Dict[str, Any]) -> None:
        # GATE: Kill-switch check (impossible to bypass)
        if should_block_execution(state):
            logger.warning(f"⛔ Execution blocked (kill switch or not running): {event_data.get('symbol', 'UNKNOWN')}")
            return  # Early return: system halted
        
        # TELEMETRY: Log signal receipt
        symbol = event_data.get('symbol', 'UNKNOWN')
        direction = event_data.get('direction', 'UNKNOWN')
        logger.info(f"Signal Received: {symbol} {direction}")
        
        # RISK CHECK: Validate with risk manager
        validated = validate_signal_with_risk_manager(risk_manager, event_data)
        
        if validated is None:
            logger.warning(f"⛔ Blocked by Risk Manager: {symbol}")
            return  # Early return: risk rejection
        
        # EXECUTION: Place order
        await executor.execute_order(validated)
        
        # STATE UPDATE: Register entry for correlation tracking
        risk_manager.register_entry(validated['symbol'])
    
    return handle_signal


# =============================================================================
# COMPONENT WIRING
# =============================================================================

def wire_detectors_to_handler(
    orchestrator: StrategyOrchestrator,
    executor: TestnetTradeExecutor,
    signal_handler: Callable
) -> None:
    """
    Wire orchestrator detectors to the signal handler.
    Side effect: Mutates detector.event_callback and registers with executor.
    """
    for symbol, detector in orchestrator.detectors.items():
        detector.event_callback = signal_handler
        executor.register_detector(symbol, detector)


def create_websocket_feed(
    active_pairs: tuple,
    orchestrator: StrategyOrchestrator
) -> BinanceWebSocketFeed:
    """
    Factory: Create configured WebSocket feed.
    No side effects until .start() is called.
    """
    return BinanceWebSocketFeed(
        symbols=list(active_pairs),
        on_1min_bar_callback=orchestrator.on_bar_update
    )


# =============================================================================
# MAIN SYSTEM CLASS
# =============================================================================

class TestnetVerificationSystem:
    """
    Testnet trading system entry point.
    
    Responsibilities:
        - Lifecycle management
        - Component initialization
        - Async task orchestration
    
    Kill-switch safety:
        - Kill switch callback stored in state, checked on every signal
        - Once triggered, no execution path can proceed
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.state = SystemState()
        self.logger.info("Initializing Testnet Verification System...")
        
        # Risk engine: Loose limits (-100R), Correlation guard (1 max)
        self.risk_manager = RiskManager(
            daily_loss_limit_r=self.state.daily_loss_limit_r,
            max_active_correlated=self.state.max_active_correlated
        )
        
        # Strategy orchestrator: Manages per-symbol detectors
        self.orchestrator = StrategyOrchestrator(
            symbols=list(self.state.active_pairs),
            max_active_trades=self.state.max_active_trades,
            enable_telegram=True
        )
        
        # Executor initialized in run() (requires bot lifecycle)
        self.executor: Optional[TestnetTradeExecutor] = None

    def _create_kill_switch_callback(self) -> Callable:
        """
        Factory: Returns kill-switch callback that mutates system state.
        
        Once triggered:
            - state.kill_switch_triggered = True
            - All subsequent signals are blocked
            - Executor's own kill-switch is also triggered
        """
        def trigger_kill_switch():
            self.state.kill_switch_triggered = True
            self.state.running = False
            self.logger.critical("🛑 KILL SWITCH TRIGGERED - All execution halted")
            if self.executor is not None:
                self.executor.trigger_kill_switch()
        
        return trigger_kill_switch

    async def run(self) -> None:
        """
        Main async entry point.
        
        Lifecycle:
            1. Start Telegram bot context
            2. Initialize executor with bot
            3. Wire kill-switch
            4. Start background tasks (polling, trailing stops)
            5. Wire signal pipeline
            6. Start WebSocket feed
            7. Await until shutdown or crash
        """
        self.state.running = True
        
        async with TelegramBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) as bot:
            # Initialize executor (requires bot for notifications)
            self.executor = TestnetTradeExecutor(telegram_bot=bot)
            
            # Wire kill-switch: Bot command -> system halt
            bot.set_kill_callback(self._create_kill_switch_callback())
            
            # Background task: Telegram polling
            poll_task = asyncio.create_task(bot.start_polling())
            
            # Background task: Trailing stop updates
            exec_task = asyncio.create_task(self.executor.update_trailing_stops())
            
            # Create signal handler with explicit dependencies
            signal_handler = create_signal_handler(
                state=self.state,
                risk_manager=self.risk_manager,
                executor=self.executor,
                logger=self.logger
            )
            
            # Wire detectors to signal pipeline
            wire_detectors_to_handler(
                orchestrator=self.orchestrator,
                executor=self.executor,
                signal_handler=signal_handler
            )
            
            # Start WebSocket feed (daemon thread)
            self.logger.info(f"Connecting to Binance Testnet Stream for {list(self.state.active_pairs)}...")
            feed = create_websocket_feed(self.state.active_pairs, self.orchestrator)
            feed.start()
            
            self.logger.info("✅ SYSTEM LIVE. Listening for Signals & /KILL command...")

            # Telegram startup notification
            pairs_str = ', '.join(self.state.active_pairs)
            await bot.send_message(
                f"🟢 <b>System Online</b>\n"
                f"Pairs: {pairs_str}\n"
                f"Feed: 1m direct (backtest-equivalent)\n"
                f"Kill: /KILL"
            )

            # Keep alive until shutdown
            try:
                await asyncio.gather(poll_task, exec_task)
            except asyncio.CancelledError:
                self.logger.info("System Shutdown Initiated")
                feed.stop()
            except Exception as e:
                self.logger.critical(f"System Crash: {e}")
                feed.stop()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    system = TestnetVerificationSystem()
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        print("Manual Stop")
