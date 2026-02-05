# telegram_bot.py
"""
Async Telegram Bot for Trade Alerts and Command Handling.

Responsibilities:
    - Send trade alerts (entry, exit, error, status)
    - Poll for commands (/KILL, /PING)
    - Execute kill-switch callback on /KILL command

Architecture:
    - BotState: Explicit, serializable state container
    - Pure functions for message formatting
    - Explicit authentication checks
    - Kill callback cannot be bypassed once triggered
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any

from bot_config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_MESSAGE_LENGTH = 4096
POLL_INTERVAL_SEC = 1
POLL_ERROR_BACKOFF_SEC = 5
LONG_POLL_TIMEOUT_SEC = 10

# Emoji mappings (frozen)
DIRECTION_EMOJIS: Dict[str, str] = {
    'LONG': '🟢',
    'SHORT': '🔴',
    'sweep': '⚡',
    'cluster': '💥',
    'thinning': '🌊'
}

# Commands
COMMAND_KILL = '/KILL'
COMMAND_PING = '/PING'


# =============================================================================
# STATE CONTAINER
# =============================================================================

@dataclass
class BotState:
    """
    Explicit, serializable bot state.
    All fields are deterministic and restart-safe.
    """
    last_update_id: int = 0
    polling_active: bool = False


# =============================================================================
# PURE FUNCTIONS: MESSAGE FORMATTING
# =============================================================================

def truncate_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Truncate message to Telegram's max length."""
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text


def format_entry_alert(
    pair: str,
    direction: str,
    entry_price: float,
    event_type: str,
    atr: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0
) -> str:
    """Format entry alert message."""
    emoji = DIRECTION_EMOJIS.get(direction, '▶️')
    msg = (
        f"<b>🎯 NEW TRADE</b>\n"
        f"{emoji} {pair} {direction}\n"
        f"Entry: {entry_price}\n"
    )
    if stop_loss > 0:
        msg += f"Stop: {stop_loss}\n"
    if take_profit > 0:
        msg += f"TP: {take_profit}\n"
    msg += (
        f"Event: {event_type}\n"
        f"ATR: {atr:.4f}"
    )
    return msg


def format_exit_alert(
    pair: str,
    direction: str,
    exit_price: float,
    pnl_bp: float,
    reason: str = "Trailing Stop"
) -> str:
    """Format exit alert message."""
    emoji = '🟢' if pnl_bp > 0 else '🔴'
    return (
        f"<b>🏁 CLOSED</b>\n"
        f"{pair} {direction}\n"
        f"Exit: {exit_price}\n"
        f"PnL: {emoji} {pnl_bp:.1f} bp\n"
        f"Reason: {reason}"
    )


def format_error_alert(error_message: str) -> str:
    """Format error alert message."""
    return f"<b>⚠️ ERROR:</b> {error_message}"


def format_status_update(message: str) -> str:
    """Format status update message."""
    return f"ℹ️ {message}"


def format_kill_received() -> str:
    """Format kill switch received message."""
    return "<b>🚨 KILL SWITCH RECEIVED. EXECUTING LOCKDOWN...</b>"


def format_kill_complete() -> str:
    """Format kill sequence complete message."""
    return "<b>✅ KILL SEQUENCE COMPLETE. SYSTEM HALTED.</b>"


def format_tsl_update(pair: str, direction: str, new_stop: float, entry_price: float) -> str:
    """Format trailing stop loss update message."""
    emoji = "🟢" if direction == "LONG" else "🔴"
    move_pct = abs((new_stop - entry_price) / entry_price * 100)
    return (f"{emoji} <b>TSL UPDATE</b> {pair}\n"
            f"Direction: {direction}\n"
            f"Entry: {entry_price:.4f}\n"
            f"New Stop: {new_stop:.4f}\n"
            f"Move: {move_pct:.2f}%")

def format_kill_failed(error: str) -> str:
    """Format kill sequence failed message."""
    return f"<b>❌ CRITICAL: KILL SEQUENCE FAILED: {error}</b>"


# =============================================================================
# PURE FUNCTIONS: VALIDATION
# =============================================================================

def is_authorized_sender(sender_id: str, authorized_chat_id: str) -> bool:
    """Check if message sender is authorized."""
    return sender_id == authorized_chat_id


def extract_command(text: str) -> str:
    """Extract and normalize command from message text."""
    return text.strip().upper()


def is_kill_command(text: str) -> bool:
    """Check if text is kill command."""
    return extract_command(text) == COMMAND_KILL


def is_ping_command(text: str) -> bool:
    """Check if text is ping command."""
    return extract_command(text) == COMMAND_PING


# =============================================================================
# PURE FUNCTIONS: UPDATE PARSING
# =============================================================================

def extract_message_info(update: Dict[str, Any]) -> tuple:
    """
    Extract message info from Telegram update.
    Returns (update_id, text, sender_id).
    """
    update_id = update.get('update_id', 0)
    
    # Check for different types of updates
    msg = update.get('message') or update.get('channel_post') or update.get('edited_message') or update.get('edited_channel_post') or {}
    
    text = msg.get('text', '').strip()
    sender_id = str(msg.get('chat', {}).get('id', ''))
    
    return update_id, text, sender_id


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_bot_logging() -> logging.Logger:
    """Configure bot logger."""
    return logging.getLogger(__name__)


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class TelegramBot:
    """
    Async Telegram bot for trade alerts and command handling.
    
    Kill-switch safety:
        - Only authorized chat_id can trigger /KILL
        - Kill callback is awaited and errors are reported
        - Kill command is logged at CRITICAL level
    
    Usage:
        async with TelegramBot(token, chat_id) as bot:
            await bot.start_polling()
    """
    
    def __init__(
        self,
        token: str = TELEGRAM_TOKEN,
        chat_id: str = TELEGRAM_CHAT_ID
    ):
        self.token = token
        self.chat_id = str(chat_id)  # Ensure string for comparison
        self.base_url = f"https://api.telegram.org/bot{token}"
        
        # HTTP session (initialized in __aenter__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State
        self.state = BotState()
        
        # Kill callback (set externally)
        self.kill_callback: Optional[Callable] = None
        
        # Logger
        self.logger = setup_bot_logging()

    # =========================================================================
    # PROPERTIES (for backward compatibility)
    # =========================================================================
    
    @property
    def last_update_id(self) -> int:
        """Backward-compatible accessor."""
        return self.state.last_update_id
    
    @last_update_id.setter
    def last_update_id(self, value: int) -> None:
        """Backward-compatible setter."""
        self.state.last_update_id = value
    
    @property
    def polling_active(self) -> bool:
        """Backward-compatible accessor."""
        return self.state.polling_active
    
    @polling_active.setter
    def polling_active(self, value: bool) -> None:
        """Backward-compatible setter."""
        self.state.polling_active = value
    
    @property
    def direction_emojis(self) -> Dict[str, str]:
        """Backward-compatible accessor for emoji mapping."""
        return DIRECTION_EMOJIS

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================
    
    async def __aenter__(self):
        """Initialize HTTP session on context entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.state.polling_active = False
        if self.session:
            await self.session.close()

    # =========================================================================
    # KILL SWITCH REGISTRATION
    # =========================================================================
    
    def set_kill_callback(self, callback: Callable) -> None:
        """
        Register the method to call when /KILL is received.
        This callback will be awaited when kill command arrives.
        """
        self.kill_callback = callback

    # =========================================================================
    # POLLING
    # =========================================================================
    
    async def start_polling(self) -> None:
        """
        Background task to poll for commands.
        Runs indefinitely until polling_active is set to False.
        """
        self.logger.info("🤖 Telegram Polling Started...")
        self.state.polling_active = True
        
        while self.state.polling_active:
            try:
                updates = await self._fetch_updates()
                for update in updates:
                    await self._process_update(update)
            except Exception as e:
                self.logger.error(f"Polling Error: {e}")
                await asyncio.sleep(POLL_ERROR_BACKOFF_SEC)
            
            await asyncio.sleep(POLL_INTERVAL_SEC)

    async def _fetch_updates(self) -> List[Dict[str, Any]]:
        """
        Fetch updates from Telegram API.
        Side effect: HTTP request.
        """
        await self._ensure_session()
        
        try:
            params = {
                'offset': self.state.last_update_id + 1,
                'timeout': LONG_POLL_TIMEOUT_SEC
            }
            async with self.session.get(f"{self.base_url}/getUpdates", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('ok'):
                        return data.get('result', [])
        except Exception:
            pass  # Errors handled in polling loop
        
        return []

    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    # =========================================================================
    # UPDATE PROCESSING
    # =========================================================================
    
    async def _process_update(self, update: Dict[str, Any]) -> None:
        """
        Process a single Telegram update.
        
        Pipeline:
            1. Extract message info
            2. Update state (last_update_id)
            3. Check authorization
            4. Route command
        """
        update_id, text, sender_id = extract_message_info(update)
        
        # STATE: Track processed update
        self.state.last_update_id = update_id
        
        # AUTH: Check sender authorization
        if not is_authorized_sender(sender_id, self.chat_id):
            return  # Early return: unauthorized sender, silently ignore
        
        # ROUTE: Handle commands
        await self._route_command(text)

    async def _route_command(self, text: str) -> None:
        """
        Route command to appropriate handler.
        """
        if is_kill_command(text):
            await self._handle_kill_command()
        elif is_ping_command(text):
            await self._handle_ping_command()
        # Unknown commands silently ignored

    async def _handle_kill_command(self) -> None:
        """
        Handle /KILL command.
        
        Sequence:
            1. Log at CRITICAL level
            2. Send acknowledgment
            3. Execute kill callback
            4. Send completion/failure message
        """
        self.logger.critical("🚨 /KILL COMMAND RECEIVED FROM AUTHORIZED USER 🚨")
        await self._send_message(format_kill_received())
        
        if self.kill_callback is None:
            self.logger.error("No Kill Callback Registered!")
            return  # Early return: no callback registered
        
        try:
            await self.kill_callback()
            await self._send_message(format_kill_complete())
        except Exception as e:
            self.logger.error(f"Kill Callback Failed: {e}")
            await self._send_message(format_kill_failed(str(e)))

    async def _handle_ping_command(self) -> None:
        """Handle /PING command."""
        await self._send_message("Pong! System Active.")

    # =========================================================================
    # MESSAGE SENDING
    # =========================================================================
    
    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send message via Telegram API.
        Side effect: HTTP request.
        Returns True on success, False on failure.
        """
        await self._ensure_session()
        
        text = truncate_message(text)
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            async with self.session.post(f"{self.base_url}/sendMessage", json=payload) as response:
                return response.status == 200
        except Exception:
            return False

    # Alias for external callers
    async def send_message(self, text: str) -> bool:
        """Public alias for _send_message."""
        return await self._send_message(text)

    # =========================================================================
    # ALERT METHODS
    # =========================================================================
    
    async def send_entry_alert(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        event_type: str,
        stop_loss: float,
        take_profit: float,
        atr: float,
        timestamp: Any,
        **kwargs
    ) -> bool:
        """
        Send entry alert.
        Note: timestamp, kwargs preserved for interface compatibility.
        """
        msg = format_entry_alert(pair, direction, entry_price, event_type, atr, stop_loss, take_profit)
        return await self._send_message(msg)

    async def send_exit_alert(
        self,
        pair: str,
        direction: str,
        exit_price: float,
        pnl_r: float,
        timestamp: Any,
        **kwargs
    ) -> bool:
        """
        Send exit alert.
        Note: timestamp, kwargs preserved for interface compatibility.
        """
        msg = format_exit_alert(pair, direction, exit_price, pnl_r)
        return await self._send_message(msg)

    async def send_tsl_update(
        self,
        pair: str,
        direction: str,
        new_stop: float,
        entry_price: float
    ) -> bool:
        """Send trailing stop loss update alert."""
        msg = format_tsl_update(pair, direction, new_stop, entry_price)
        return await self._send_message(msg)

    async def send_error_alert(self, error_message: str) -> bool:
        """Send error alert."""
        return await self._send_message(format_error_alert(error_message))

    async def send_status_update(self, message: str) -> bool:
        """Send status update."""
        return await self._send_message(format_status_update(message))