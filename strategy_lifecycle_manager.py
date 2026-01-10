# strategy_lifecycle_manager.py
"""
Fully automated strategy lifecycle manager
- Weekly self-assessment every Sunday 23:00 UTC
- Auto-apply corrections if decay > 0.70
- Auto-pause if decay > 0.85 for 3 consecutive weeks
- Archives strategy if terminal
"""

import asyncio
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from scipy import stats
import logging

class StrategyLifecycleManager:
    def __init__(self, trade_log_path: str, telegram_bot: Any):
        self.trade_log_path = trade_log_path
        self.telegram = telegram_bot
        
        # State persistence
        self.state_file = 'data/system/strategy_state.json'
        os.makedirs('data/system', exist_ok=True)
        
        # Load historical state
        self.state = self._load_state()
        
        # Baseline from Dec 2025 validation
        self.BASELINE = {
            'avg_pnl_bp': 7.51,
            'win_rate': 0.414,
            'trades_per_week': 75
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _load_state(self) -> Dict[str, Any]:
        """Load persistent state from disk"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'decay_scores': [],
            'corrections_applied': [],
            'is_paused': False,
            'archived': False,
            'consecutive_critical_weeks': 0,
            'last_check': None
        }
    
    def _save_state(self):
        """Atomic write to state file"""
        temp_file = self.state_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        os.replace(temp_file, self.state_file)
    
    async def run_weekly_check(self):
        """Execute weekly strategy health assessment"""
        now = datetime.utcnow()
        
        # Only run on Sundays at 23:00 UTC
        if now.weekday() != 6 or now.hour != 23:
            return
        
        self.logger.info(f"🔄 Running weekly strategy health check for week of {now.strftime('%Y-W%U')}")
        
        # Load trade data
        if not os.path.exists(self.trade_log_path):
            self.logger.warning("No trade log found - skipping check")
            return
        
        trades = pd.read_csv(self.trade_log_path, parse_dates=['timestamp'])
        cutoff_date = now - timedelta(days=30)
        recent_trades = trades[trades['timestamp'] >= cutoff_date]
        
        if len(recent_trades) < 30:
            self.logger.info(f"Insufficient data: {len(recent_trades)} trades (< 30)")
            return
        
        # Calculate decay metrics
        decay_type, decay_score = self._calculate_decay_metrics(recent_trades)
        
        # Store result
        self.state['decay_scores'].append({
            'week': now.strftime('%Y-W%U'),
            'score': decay_score,
            'type': decay_type,
            'timestamp': now.isoformat()
        })
        
        # Decision logic
        if decay_score >= 0.85:
            await self._handle_critical_decay(decay_type, decay_score, now)
        elif decay_score >= 0.70:
            await self._handle_moderate_decay(decay_type, decay_score, now)
        else:
            await self._handle_healthy_state(decay_score, now)
        
        # Cleanup old scores (keep last 12 weeks)
        self.state['decay_scores'] = self.state['decay_scores'][-12:]
        
        self._save_state()
        self.state['last_check'] = now.isoformat()
    
    def _calculate_decay_metrics(self, trades: pd.DataFrame) -> tuple:
        """Calculate four-factor decay score"""
        if len(trades) < 30:
            return "insufficient_data", 0.0
        
        # Signal 1: Win-rate decay
        current_wr = (trades['pnl_bp'] > 0).mean()
        wr_decay = max(0, self.BASELINE['win_rate'] - current_wr) / self.BASELINE['win_rate']
        
        # Signal 2: PnL significance
        recent_pnl = trades['pnl_bp'].iloc[-15:]
        t_stat, p_val = stats.ttest_1samp(recent_pnl, 0.0)
        pnl_decay = 1.0 if p_val > 0.10 else 0.0
        
        # Signal 3: Alpha compression
        current_pnl = trades['pnl_bp'].mean()
        compression_decay = max(0, self.BASELINE['avg_pnl_bp'] - current_pnl) / self.BASELINE['avg_pnl_bp']
        
        # Signal 4: Drawdown severity
        running_pnl = trades['pnl_bp'].cumsum()
        max_dd = (running_pnl - running_pnl.expanding().max()).min()
        dd_decay = 1.0 if max_dd < -200 else 0.0
        
        # Composite score
        decay_score = np.mean([wr_decay, pnl_decay, compression_decay, dd_decay])
        
        # Classify type
        if wr_decay > 0.50:
            dtype = "win_rate_collapse"
        elif pnl_decay > 0.50:
            dtype = "edge_not_significant"
        elif compression_decay > 0.50:
            dtype = "alpha_compression"
        elif dd_decay == 1.0:
            dtype = "max_drawdown_exceeded"
        else:
            dtype = "healthy"
        
        return dtype, decay_score
    
    async def _handle_critical_decay(self, decay_type: str, score: float, now: datetime):
        """Score >= 0.85: Pause system and alert"""
        self.state['consecutive_critical_weeks'] += 1
        
        self.logger.critical(f"CRITICAL DECAY ({score:.2f}): {decay_type} - PAUSING")
        
        self.state['is_paused'] = True
        
        # After 3 consecutive critical weeks, archive
        if self.state['consecutive_critical_weeks'] >= 3:
            self.state['archived'] = True
            self.logger.critical("STRATEGY TERMINAL - ARCHIVING")
            action_msg = "ARCHIVED - Research v2 strategy"
        else:
            action_msg = "PAUSED - Manual intervention required"
        
        message = (
            f"🚨 <b>STRATEGY CRITICAL DECAY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Score:</b> {score:.2f}\n"
            f"🔍 <b>Type:</b> {decay_type}\n"
            f"📅 <b>Consecutive Critical Weeks:</b> {self.state['consecutive_critical_weeks']}\n"
            f"⏸️ <b>Status:</b> {action_msg}\n"
            f"📋 <b>Trade Log:</b> {self.trade_log_path}"
        )
        await self.telegram.send_error_alert(message)
    
    async def _handle_moderate_decay(self, decay_type: str, score: float, now: datetime):
        """Score 0.70-0.85: Apply one correction"""
        self.logger.warning(f"MODERATE DECAY ({score:.2f}): {decay_type} - APPLYING CORRECTION")
        self.state['consecutive_critical_weeks'] = 0  # Reset
        
        correction = self._select_correction(decay_type)
        
        if correction:
            await self._apply_correction(correction)
            
            self.state['corrections_applied'].append({
                'week': now.strftime('%Y-W%U'),
                'type': decay_type,
                'correction': correction,
                'score_before': score,
                'validated': False
            })
            
            message = (
                f"🔧 <b>DECAY CORRECTION APPLIED</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Score:</b> {score:.2f}\n"
                f"🔍 <b>Type:</b> {decay_type}\n"
                f"🔧 <b>Correction:</b> {correction['name']}\n"
                f"✅ <b>Status:</b> Monitoring validation for 7 days\n"
                f"📝 <b>Live Params:</b> data/system/live_params.json"
            )
            await self.telegram.send_status_update(message)
    
    async def _handle_healthy_state(self, score: float, now: datetime):
        """Score < 0.70: Normal operation"""
        self.state['consecutive_critical_weeks'] = 0
        
        # Check if last correction was effective
        if self.state['corrections_applied']:
            last_corr = self.state['corrections_applied'][-1]
            if not last_corr.get('validated', False):
                if self._validate_last_correction():
                    last_corr['validated'] = True
                    message = "✅ Last correction validated - performance improved"
                else:
                    message = "❌ Last correction ineffective - reverting next week"
                    await self._revert_last_correction()
                
                await self.telegram.send_status_update(message)
        
        self.logger.info(f"Strategy healthy (score={score:.2f})")
    
    def _select_correction(self, decay_type: str) -> Dict[str, Any]:
        """Map decay type to correction action"""
        return {
            "win_rate_collapse": {
                'name': 'Increase Entry Delay & Filter',
                'params': {
                    'cooldown_minutes': 30,
                    'min_atr': 0.20,
                    'volume_confirmation': 1.3
                }
            },
            "alpha_compression": {
                'name': 'Reduce Trade Frequency',
                'params': {
                    'max_trades_per_day': 4,
                    'volume_confirmation': 1.4,
                    'min_atr': 0.18
                }
            },
            "edge_not_significant": {
                'name': 'Full Parameter Re-optimization',
                'params': {}  # Triggers offline backtest
            }
        }.get(decay_type)
    
    async def _apply_correction(self, correction: Dict[str, Any]):
        """Write correction to live params file"""
        os.makedirs('data/system', exist_ok=True)
        with open('data/system/live_params.json', 'w') as f:
            json.dump(correction['params'], f, indent=2)
        
        self.logger.info(f"Applied correction: {correction['name']}")
    
    def _validate_last_correction(self) -> bool:
        """Check if last correction improved score by >= 0.15"""
        if len(self.state['decay_scores']) < 2:
            return False
        
        before = self.state['decay_scores'][-2]['score']
        after = self.state['decay_scores'][-1]['score']
        
        return (before - after) >= 0.15
    
    async def _revert_last_correction(self):
        """Revert to baseline parameters"""
        baseline_path = 'config/trade_params_baseline.json'
        if os.path.exists(baseline_path):
            import shutil
            shutil.copy(baseline_path, 'data/system/live_params.json')
            self.logger.info("Reverted to baseline parameters")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_paused': self.state['is_paused'],
            'archived': self.state['archived'],
            'last_score': self.state['decay_scores'][-1]['score'] if self.state['decay_scores'] else 0.0,
            'corrections_active': len(self.state['corrections_applied'])
        }

# Scheduler function
async def run_lifecycle_scheduler(manager: StrategyLifecycleManager):
    """Run weekly checks every Sunday 23:00 UTC"""
    while True:
        now = datetime.utcnow()
        
        # Calculate next Sunday 23:00
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 23:
            days_until_sunday = 7
        
        next_check = now + timedelta(days=days_until_sunday)
        next_check = next_check.replace(hour=23, minute=0, second=0, microsecond=0)
        
        wait_seconds = (next_check - now).total_seconds()
        manager.logger.info(f"Next lifecycle check: {next_check} (in {wait_seconds/3600:.1f}h)")
        
        await asyncio.sleep(wait_seconds)
        await manager.run_weekly_check()