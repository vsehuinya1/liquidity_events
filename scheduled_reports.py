# scheduled_reports.py
"""
Scheduled Performance Reports for Telegram

Sends daily and weekly summaries at configured times.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, time as dt_time, timedelta
from typing import Optional
import os


# Report schedule (UTC times)
DAILY_REPORT_TIME = dt_time(20, 0)  # 20:00 UTC = 23:00 Nairobi
WEEKLY_REPORT_TIME = dt_time(17, 0)  # 17:00 UTC Sunday = 20:00 Nairobi Sunday
WEEKLY_REPORT_DAY = 6  # Sunday (0=Monday, 6=Sunday)

METRIC_LOG_PATH = 'analysis/trade_metrics.csv'  # BUG FIX 3: Read from trade_metrics.csv (exits), not verify_latency.csv (entries)
STATE_FILE = 'data/state/execution_state.json'


def setup_report_logging() -> logging.Logger:
    return logging.getLogger('ScheduledReports')


def load_daily_metrics() -> pd.DataFrame:
    """Load today's trade metrics."""
    if not os.path.exists(METRIC_LOG_PATH):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(METRIC_LOG_PATH)
        if df.empty:
            return df
        
        # Parse timestamp and filter for today (BUG FIX 3: Use 'timestamp' column from trade_metrics.csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today = datetime.utcnow().date()
        df_today = df[df['timestamp'].dt.date == today]
        return df_today
    except Exception as e:
        logging.error(f"Failed to load daily metrics: {e}")
        return pd.DataFrame()


def load_weekly_metrics() -> pd.DataFrame:
    """Load this week's trade metrics."""
    if not os.path.exists(METRIC_LOG_PATH):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(METRIC_LOG_PATH)
        if df.empty:
            return df
        
        # Parse timestamp and filter for this week (BUG FIX 3: Use 'timestamp' column from trade_metrics.csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today = datetime.utcnow()
        week_start = today - timedelta(days=today.weekday())
        df_week = df[df['timestamp'] >= week_start]
        return df_week
    except Exception as e:
        logging.error(f"Failed to load weekly metrics: {e}")
        return pd.DataFrame()


def format_daily_report(df: pd.DataFrame) -> str:
    """Format daily performance summary."""
    if df.empty:
        return "📊 <b>Daily Summary</b>\n\nNo trades today."
    
    total_trades = len(df)
    
    # BUG FIX 3: Use trade_metrics.csv columns (pnl_r, symbol)
    total_pnl_r = df['pnl_r'].sum() if 'pnl_r' in df.columns else 0
    wins = len(df[df['pnl_r'] > 0]) if 'pnl_r' in df.columns else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    today = datetime.utcnow().strftime('%b %d')
    
    msg = (
        f"📊 <b>Daily Summary - {today}</b>\n\n"
        f"Trades: {total_trades}\n"
        f"Total PnL: {total_pnl_r:+.2f}R\n"
        f"Win Rate: {win_rate:.1f}%\n"
    )
    
    # Show symbols traded
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        msg += f"Pairs: {', '.join(symbols)}\n"
    
    return msg


def format_weekly_report(df: pd.DataFrame) -> str:
    """Format weekly performance summary."""
    if df.empty:
        return "📈 <b>Weekly Summary</b>\n\nNo trades this week."
    
    total_trades = len(df)
    
    # BUG FIX 3: Use trade_metrics.csv columns (pnl_r, symbol, timestamp)
    total_pnl_r = df['pnl_r'].sum() if 'pnl_r' in df.columns else 0
    wins = len(df[df['pnl_r'] > 0]) if 'pnl_r' in df.columns else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Group by day
    daily_counts = df.groupby(df['timestamp'].dt.date).size()
    busiest_day = daily_counts.idxmax() if not daily_counts.empty else None
    
    week_num = datetime.utcnow().isocalendar()[1]
    
    msg = (
        f"📈 <b>Week {week_num} Performance</b>\n\n"
        f"Total Trades: {total_trades}\n"
        f"Total PnL: {total_pnl_r:+.2f}R\n"
        f"Win Rate: {win_rate:.1f}%\n"
    )
    
    if busiest_day:
        msg += f"Busiest Day: {busiest_day} ({daily_counts[busiest_day]} trades)\n"
    
    # Show symbols traded
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        msg += f"Pairs: {', '.join(symbols)}\n"
    
    return msg


async def send_daily_report(telegram_bot) -> None:
    """Send daily performance report."""
    logger = setup_report_logging()
    logger.info("Generating daily report...")
    
    df = load_daily_metrics()
    report = format_daily_report(df)
    
    try:
        await telegram_bot.send_message(report)
        logger.info("Daily report sent")
    except Exception as e:
        logger.error(f"Failed to send daily report: {e}")


async def send_weekly_report(telegram_bot) -> None:
    """Send weekly performance report."""
    logger = setup_report_logging()
    logger.info("Generating weekly report...")
    
    df = load_weekly_metrics()
    report = format_weekly_report(df)
    
    try:
        await telegram_bot.send_message(report)
        logger.info("Weekly report sent")
    except Exception as e:
        logger.error(f"Failed to send weekly report: {e}")


def seconds_until_time(target_time: dt_time) -> int:
    """Calculate seconds until next occurrence of target time."""
    now = datetime.utcnow()
    target = datetime.combine(now.date(), target_time)
    
    if target <= now:
        target += timedelta(days=1)
    
    return int((target - now).total_seconds())


def seconds_until_weekly_report() -> int:
    """Calculate seconds until next weekly report (Sunday at target time)."""
    now = datetime.utcnow()
    target_time = WEEKLY_REPORT_TIME
    
    # Calculate days until Sunday
    days_ahead = WEEKLY_REPORT_DAY - now.weekday()
    if days_ahead < 0:  # Already past this week's Sunday
        days_ahead += 7
    elif days_ahead == 0 and now.time() >= target_time:  # Today is Sunday but past report time
        days_ahead = 7
    
    target = datetime.combine(now.date() + timedelta(days=days_ahead), target_time)
    return int((target - now).total_seconds())


async def daily_report_task(telegram_bot) -> None:
    """Background task for daily reports."""
    logger = setup_report_logging()
    logger.info("Daily report task started")
    
    last_send_date = None
    
    while True:
        try:
            # Wait until next daily report time
            wait_seconds = seconds_until_time(DAILY_REPORT_TIME)
            
            # Don't send if we already sent today (prevents duplicates on restart)
            today = datetime.utcnow().date()
            if last_send_date == today:
                wait_seconds = max(wait_seconds, 3600)  # Wait at least an hour
            
            logger.info(f"Next daily report in {wait_seconds}s ({wait_seconds/3600:.1f}h)")
            await asyncio.sleep(wait_seconds)
            
            # Double-check we haven't sent today
            today = datetime.utcnow().date()
            if last_send_date == today:
                continue
            
            # Send report
            await send_daily_report(telegram_bot)
            last_send_date = today
            
            # Wait 1 minute to avoid any edge cases
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Daily report task error: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour on error


async def weekly_report_task(telegram_bot) -> None:
    """Background task for weekly reports."""
    logger = setup_report_logging()
    logger.info("Weekly report task started")
    
    while True:
        try:
            # Wait until next weekly report time
            wait_seconds = seconds_until_weekly_report()
            logger.info(f"Next weekly report in {wait_seconds}s ({wait_seconds/3600:.1f}h)")
            await asyncio.sleep(wait_seconds)
            
            # Send report
            await send_weekly_report(telegram_bot)
            
        except Exception as e:
            logger.error(f"Weekly report task error: {e}")
            await asyncio.sleep(3600)  # Retry in 1 hour on error
