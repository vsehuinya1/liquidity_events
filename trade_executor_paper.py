# trade_executor_paper.py
"""
Paper Trade Executor with live parameter loading and decay-aware risk management
"""

import asyncio
import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

class PaperTradeExecutor:
    def __init__(self, telegram_bot: Any, params_path: str = 'config/trade_params_2026_enhanced.json'):
        self.telegram = telegram_bot
        
        # Load parameters (will be auto-reloaded from live_params.json if corrections applied)
        self.params_path = params_path
        self.params = self._load_params()
        
        # State
        self.today_trades = []
        self.daily_pnl = 0.0
        self.current_date = datetime.utcnow().date()
        
        # Trade tracking
        os.makedirs('data/trades', exist_ok=True)
        self.trade_log_path = f"data/trades/paper_trades_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        
        # Ensure CSV exists
        if not os.path.exists(self.trade_log_path):
            pd.DataFrame(columns=[
                'timestamp', 'direction', 'entry_price', 'exit_price', 
                'pnl_bp', 'reason', 'hold_time', 'event_type', 'decay_score_at_entry'
            ]).to_csv(self.trade_log_path, index=False)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Paper executor initialized with params: {self.params}")
    
    def _load_params(self) -> Dict[str, Any]:
        """Load params from file (auto-reloads if changes detected)"""
        os.makedirs('data/system', exist_ok=True)
        
        if os.path.exists('data/system/live_params.json'):
            try:
                with open('data/system/live_params.json', 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.logger.warning("Corrupted or missing live_params.json, using baseline")
        
        # Fallback to baseline
        with open(self.params_path, 'r') as f:
            baseline_params = json.load(f)
        
        # Create live_params.json if it doesn't exist
        with open('data/system/live_params.json', 'w') as f:
            json.dump(baseline_params, f, indent=2)
        
        return baseline_params
    
    def reload_params(self):
        """Call this after lifecycle manager applies correction"""
        self.params = self._load_params()
        self.logger.info(f"Reloaded params: {self.params}")
    
    async def on_event(self, event: Dict[str, Any]):
        """Called by LiveEventDetector when event fires"""
        
        # Check decay filter
        if event.get('decay_score', 0.0) >= self.params.get('decay_threshold_realtime', 0.75):
            self.logger.info(f"Trade skipped: decay score {event.get('decay_score', 0.0):.2f}")
            return
        
        # Check daily limits
        if len(self.today_trades) >= self.params['max_trades_per_day']:
            self.logger.info("Daily trade limit reached")
            return
        
        if self.daily_pnl <= self.params.get('daily_loss_limit_bp', -150):
            self.logger.info("Daily loss limit hit")
            return
        
        # Simulate trade execution
        await self._simulate_trade(event)
    
    async def _simulate_trade(self, event: Dict[str, Any]):
        """Simulate full trade lifecycle with realistic slippage"""
        
        # Determine direction
        direction_map = {
            'liquidity_sweep': 'SHORT' if event.get('direction') == 'UPPER' else 'LONG',
            'liquidation_cluster': 'SHORT' if event.get('direction') == 'BULLISH' else 'LONG',
            'liquidity_thinning': 'LONG'
        }
        direction = direction_map.get(event['event_type'])
        if not direction:
            return
        
        # Entry parameters
        entry_price = float(event['bar']['close'])
        atr = float(event['atr'])
        
        # Calculate risk levels
        stop_loss = entry_price - (atr * self.params['stop_loss_atr_mult']) * (1 if direction == 'LONG' else -1)
        take_profit = entry_price + (atr * self.params['take_profit_atr_mult']) * (1 if direction == 'LONG' else -1)
        
        entry_time = datetime.utcnow()
        
        # Log to CSV (simulating entry)
        trade_record = {
            'timestamp': entry_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': None,
            'pnl_bp': None,
            'reason': 'open',
            'hold_time': 0,
            'event_type': event['event_type'],
            'decay_score_at_entry': event.get('decay_score', 0.0)
        }
        
        pd.DataFrame([trade_record]).to_csv(self.trade_log_path, mode='a', header=False, index=False)
        
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
        
        # Simulate hold period (random from params)
        import random
        hold_time = random.choice(self.params['hold_times'])
        await asyncio.sleep(hold_time * 60)  # Convert to seconds
        
        # Simulate exit with slippage
        slippage_factor = 0.0002  # 0.2 bp slippage
        if direction == 'LONG':
            exit_price = entry_price * (1 + slippage_factor + (atr * 2.5 / entry_price))
        else:
            exit_price = entry_price * (1 - slippage_factor - (atr * 2.5 / entry_price))
        
        # Calculate PnL
        raw_return = (exit_price - entry_price) / entry_price * 10000 * (1 if direction == 'LONG' else -1)
        net_return = raw_return - (2 * self.params['transaction_cost'] * 10000)
        pnl_bp = net_return
        
        exit_time = datetime.utcnow()
        
        # Log exit
        trade_record.update({
            'exit_price': exit_price,
            'pnl_bp': pnl_bp,
            'reason': 'simulated_exit',
            'hold_time': hold_time
        })
        
        # Update CSV (mark as closed)
        df = pd.read_csv(self.trade_log_path)
        df.loc[df['reason'] == 'open', ['exit_price', 'pnl_bp', 'reason', 'hold_time']] = [
            exit_price, pnl_bp, 'simulated_exit', hold_time
        ]
        df.to_csv(self.trade_log_path, index=False)
        
        # Send exit alert
        await self.telegram.send_exit_alert(
            pair="SOLUSDT",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_bp=pnl_bp,
            reason="simulated_exit",
            hold_time=hold_time,
            timestamp=exit_time
        )
        
        # Update daily state
        self.today_trades.append(trade_record)
        self.daily_pnl += pnl_bp
        
        self.logger.info(f"Paper trade: {direction} at {entry_price:.4f}, P&L: {pnl_bp:.1f} bp")
    
    async def finalize(self):
        """Send daily summary at shutdown"""
        if not self.today_trades:
            return
        
        df = pd.DataFrame(self.today_trades)
        
        # Load full history for summary
        full_df = pd.read_csv(self.trade_log_path)
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        today = datetime.utcnow().date()
        today_trades = full_df[full_df['timestamp'].dt.date == today]
        
        if today_trades.empty:
            return
        
        # Calculate metrics
        total_pnl = today_trades['pnl_bp'].sum()
        win_rate = (today_trades['pnl_bp'] > 0).mean()
        max_dd = (today_trades['pnl_bp'].cumsum() - today_trades['pnl_bp'].cumsum().expanding().max()).min()
        
        await self.telegram.send_daily_summary(
            date=today.strftime('%Y-%m-%d'),
            total_trades=len(today_trades),
            net_pnl_bp=total_pnl,
            win_rate=win_rate,
            max_dd_bp=max_dd,
            avg_hold_time=today_trades['hold_time'].mean(),
            best_trade_bp=today_trades['pnl_bp'].max(),
            worst_trade_bp=today_trades['pnl_bp'].min(),
            trades_by_event=today_trades['event_type'].value_counts().to_dict()
        )