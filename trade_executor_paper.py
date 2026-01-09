# trade_executor_paper.py
"""
Paper Trade Executor - Logs trades without real API calls
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import os

class PaperTradeExecutor:
    """Logs paper trades to file and sends Telegram alerts"""
    
    def __init__(self, telegram, max_trades_per_day: int = 5, daily_loss_limit_bp: float = -150):
        self.telegram = telegram
        self.max_trades_per_day = max_trades_per_day
        self.daily_loss_limit_bp = daily_loss_limit_bp
        
        # State
        self.today_trades = []
        self.daily_pnl = 0.0
        self.current_date = datetime.utcnow().date()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Trade log file
        os.makedirs('data/trades', exist_ok=True)
        self.trade_log_file = 'data/trades/paper_trades.csv'
        
        # Ensure file exists with headers
        if not os.path.exists(self.trade_log_file):
            pd.DataFrame(columns=[
                'timestamp', 'direction', 'entry_price', 'exit_price', 
                'pnl_bp', 'reason', 'hold_time', 'event_type'
            ]).to_csv(self.trade_log_file, index=False)
    
    def on_event(self, event: Dict[str, Any]):
        """Called by LiveEventDetector when event fires"""
        
        # Check for new day rollover
        current_date = datetime.utcnow().date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.today_trades = []
            self.daily_pnl = 0.0
            self.logger.info(f"New trading day: {current_date}")
        
        # Schedule async trade simulation on event loop
        asyncio.create_task(self._simulate_trade(event))
    
    async def _simulate_trade(self, event: Dict[str, Any]):
        """Simulate a paper trade based on event"""
        
        # Check daily limits
        if len(self.today_trades) >= self.max_trades_per_day:
            self.logger.info("Daily trade limit reached, skipping")
            return
        
        if self.daily_pnl <= self.daily_loss_limit_bp:
            self.logger.info("Daily loss limit hit, stopping for today")
            return
        
        # Determine trade direction
        direction_map = {
            'liquidity_sweep': 'SHORT' if event.get('direction') == 'UPPER' else 'LONG',
            'liquidation_cluster': 'SHORT' if event.get('direction') == 'BULLISH' else 'LONG',
            'liquidity_thinning': 'SHORT' if event['bar']['close'] > event['bar']['open'] else 'LONG'
        }
        
        direction = direction_map.get(event['event_type'])
        if not direction:
            self.logger.error(f"Unknown event type: {event['event_type']}")
            return
        
        # Simulate entry
        entry_price = float(event['bar']['close'])
        atr = float(event['atr'])
        
        # Risk parameters (3:1 RR)
        stop_loss = entry_price - (atr * 1.5) * (1 if direction == 'LONG' else -1)
        take_profit = entry_price + (atr * 4.5) * (1 if direction == 'LONG' else -1)
        
        entry_time = datetime.utcnow()
        
        # Send entry alert
        await self.telegram.send_entry_alert(
            pair="SOLUSDT",
            direction=direction,
            entry_price=entry_price,
            event_type=event['event_type'],
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            timestamp=entry_time
        )
        
        # Simulate hold and exit (for demo - in reality use price monitoring)
        await asyncio.sleep(30)  # Simulate 30-second hold
        
        # Simulate exit
        exit_price = entry_price + (atr * 2 * (1 if direction == 'LONG' else -1))
        pnl_bp = 15.0  # Simulated profit
        
        await self.telegram.send_exit_alert(
            pair="SOLUSDT",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_bp=pnl_bp,
            reason="take_profit",
            hold_time=1,
            timestamp=datetime.utcnow()
        )
        
        # Log trade
        trade_record = {
            'timestamp': entry_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_bp': pnl_bp,
            'reason': 'take_profit',
            'hold_time': 1,
            'event_type': event['event_type']
        }
        
        # Append to CSV
        pd.DataFrame([trade_record]).to_csv(
            self.trade_log_file, 
            mode='a', 
            header=False, 
            index=False
        )
        
        self.today_trades.append(trade_record)
        self.daily_pnl += pnl_bp
        
        self.logger.info(f"Paper trade: {direction} at {entry_price:.4f}, P&L: {pnl_bp:.1f} bp")
    
    async def finalize(self):
        """Send daily summary at shutdown"""
        if not self.today_trades:
            return
        
        df = pd.DataFrame(self.today_trades)
        
        min_date = df['timestamp'].dt.date.min()
        
        await self.telegram.send_daily_summary(
            date=min_date.strftime('%Y-%m-%d'),
            total_trades=len(self.today_trades),
            net_pnl_bp=self.daily_pnl,
            win_rate=(df['pnl_bp'] > 0).mean(),
            max_dd_bp=df['pnl_bp'].cumsum().min(),
            avg_hold_time=df['hold_time'].mean(),
            best_trade_bp=df['pnl_bp'].max(),
            worst_trade_bp=df['pnl_bp'].min(),
            trades_by_event=df['event_type'].value_counts().to_dict()
        )