# telegram_bot.py
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    """
    Async Telegram bot for trading alerts
    - Entry/exit notifications
    - Daily summary reports
    - Error notifications
    """
    
    def __init__(self, token: str = TELEGRAM_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Emoji mapping for visual alerts
        self.direction_emojis = {
            'LONG': '🟢',
            'SHORT': '🔴',
            'sweep': '⚡',
            'cluster': '💥',
            'thinning': '🌊'
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Internal method to send message via Telegram API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Truncate if too long (Telegram limit: 4096 chars)
        if len(text) > 4096:
            text = text[:4093] + "..."
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    self.logger.info(f"Telegram message sent: {text[:50]}...")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Telegram API error {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    # =========================================================================
    # ALERT TYPE 1: ENTRY ALERT
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
        timestamp: datetime
    ) -> bool:
        """
        Entry alert: pair, direction, price, event, SL, TP
        """
        emoji = self.direction_emojis.get(direction, '▶️')
        event_emoji = self.direction_emojis.get(event_type, '📊')
        
        message = (
            f"<b>🎯 NEW TRADE SIGNAL</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} <b>Pair:</b> {pair}\n"
            f"{emoji} <b>Direction:</b> {direction}\n"
            f"💰 <b>Entry Price:</b> <code>{entry_price:.4f}</code>\n"
            f"{event_emoji} <b>Event Type:</b> {event_type.upper()}\n"
            f"📊 <b>ATR:</b> {atr:.4f}\n"
            f"🕐 <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🛑 <b>Stop Loss:</b> <code>{stop_loss:.4f}</code>\n"
            f"🎯 <b>Take Profit:</b> <code>{take_profit:.4f}</code>\n"
            f"📏 <b>Risk/Reward:</b> 1:3\n"
        )
        
        return await self._send_message(message)
    
    # =========================================================================
    # ALERT TYPE 2: EXIT ALERT
    # =========================================================================
    async def send_exit_alert(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_bp: float,
        reason: str,
        hold_time: int,
        timestamp: datetime
    ) -> bool:
        """
        Exit alert: PnL, reason, hold time
        """
        emoji = self.direction_emojis.get(direction, '▶️')
        
        # PnL coloring
        if pnl_bp > 0:
            pnl_text = f"+{pnl_bp:.1f} bp ✅"
            pnl_color = "🟢"
        elif pnl_bp < 0:
            pnl_text = f"{pnl_bp:.1f} bp ❌"
            pnl_color = "🔴"
        else:
            pnl_text = "0.0 bp"
            pnl_color = "⚪"
        
        # Exit reason emoji
        reason_emojis = {
            'stop_loss': '🛑',
            'take_profit': '🎯',
            'time': '⏱️'
        }
        reason_emoji = reason_emojis.get(reason, '🚪')
        
        # Calculate raw return
        raw_return = (exit_price - entry_price) / entry_price * 10000 * (1 if direction == 'LONG' else -1)
        
        message = (
            f"<b>🏁 TRADE CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} <b>Pair:</b> {pair}\n"
            f"{emoji} <b>Direction:</b> {direction}\n"
            f"💰 <b>Entry:</b> <code>{entry_price:.4f}</code>\n"
            f"💰 <b>Exit:</b> <code>{exit_price:.4f}</code>\n"
            f"{pnl_color} <b>P&L:</b> {pnl_text}\n"
            f"📊 <b>Raw Return:</b> {raw_return:+.1f} bp\n"
            f"⏱️ <b>Hold Time:</b> {hold_time} minutes\n"
            f"{reason_emoji} <b>Exit Reason:</b> {reason.replace('_', ' ').title()}\n"
            f"🕐 <b>Close Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        )
        
        return await self._send_message(message)
    
    # =========================================================================
    # ALERT TYPE 3: DAILY SUMMARY
    # =========================================================================
    async def send_daily_summary(
        self,
        date: str,
        total_trades: int,
        net_pnl_bp: float,
        win_rate: float,
        max_dd_bp: float,
        avg_hold_time: float,
        best_trade_bp: float,
        worst_trade_bp: float,
        trades_by_event: Dict[str, int]
    ) -> bool:
        """
        Daily summary: trades, net PnL, max DD (22:00 UTC)
        """
        # Overall performance emoji
        if net_pnl_bp >= 100:
            performance_emoji = "🚀"
        elif net_pnl_bp >= 50:
            performance_emoji = "📈"
        elif net_pnl_bp >= 0:
            performance_emoji = "✅"
        elif net_pnl_bp >= -50:
            performance_emoji = "📉"
        else:
            performance_emoji = "⚠️"
        
        # Max DD coloring
        dd_text = f"{max_dd_bp:.1f} bp" if max_dd_bp < 0 else "0.0 bp"
        
        # Event breakdown
        event_lines = "\n".join([
            f"   {event}: {count} trades"
            for event, count in trades_by_event.items()
        ])
        
        message = (
            f"<b>{performance_emoji} DAILY SUMMARY - {date}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Total Trades:</b> {total_trades}\n"
            f"💰 <b>Net P&L:</b> <code>{net_pnl_bp:+.1f} bp</code>\n"
            f"🏆 <b>Win Rate:</b> {win_rate:.1%}\n"
            f"📉 <b>Max Drawdown:</b> {dd_text}\n"
            f"⏱️ <b>Avg Hold Time:</b> {avg_hold_time:.1f} min\n"
            f"🎯 <b>Best Trade:</b> +{best_trade_bp:.1f} bp\n"
            f"💥 <b>Worst Trade:</b> {worst_trade_bp:.1f} bp\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Trades by Event Type:</b>\n"
            f"{event_lines}\n"
        )
        
        return await self._send_message(message)
    
    # =========================================================================
    # UTILITY ALERTS
    # =========================================================================
    async def send_error_alert(self, error_message: str) -> bool:
        """Critical error notification"""
        message = (
            f"<b>⚠️ SYSTEM ERROR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<code>{error_message}</code>\n"
            f"🕐 <b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        )
        return await self._send_message(message)
    
    async def send_status_update(self, message: str) -> bool:
        """General status update"""
        return await self._send_message(f"<b>ℹ️ Status Update</b>\n━━━━━━━━━━━━━━━━━━━━━━\n{message}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demo_alerts():
    """Demonstrate all three alert types"""
    
    async with TelegramBot() as bot:
        # 1. Entry Alert
        await bot.send_entry_alert(
            pair="SOLUSDT",
            direction="LONG",
            entry_price=195.5678,
            event_type="sweep",
            stop_loss=194.5678,
            take_profit=199.5678,
            atr=0.8923,
            timestamp=datetime.utcnow()
        )
        
        await asyncio.sleep(2)
        
        # 2. Exit Alert
        await bot.send_exit_alert(
            pair="SOLUSDT",
            direction="LONG",
            entry_price=195.5678,
            exit_price=199.1234,
            pnl_bp=18.4,
            reason="take_profit",
            hold_time=12,
            timestamp=datetime.utcnow()
        )
        
        await asyncio.sleep(2)
        
        # 3. Daily Summary
        await bot.send_daily_summary(
            date="2025-01-09",
            total_trades=4,
            net_pnl_bp=45.2,
            win_rate=0.75,
            max_dd_bp=-12.8,
            avg_hold_time=18.5,
            best_trade_bp=28.4,
            worst_trade_bp=-8.2,
            trades_by_event={
                "sweep": 2,
                "cluster": 1,
                "thinning": 1
            }
        )


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_alerts())