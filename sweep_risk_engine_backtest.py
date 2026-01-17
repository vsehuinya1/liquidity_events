# sweep_risk_engine_backtest.py

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# =============================
# GLOBAL CONFIG
# =============================

DATA_PATH = 'data/parquet/SOLUSDT_1m_with_funding.parquet'
INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5  # round-trip
ATR_PERIOD = 20

# Containment Config (v1.1.0)
MAX_DRAWDOWN_SESSION_R = 2.0
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_TRADES = 5
REGIME_WINDOW = 10

# RISK ENGINE CONFIG (Phase 2)
DAILY_LOSS_LIMIT_R = -3.0
ENABLE_SESSION_SCALING = True

# Statistics
stats = {
    'attack_activations': 0,
    'attack_kills_dd': 0,
    'attack_kills_loss': 0,
    'attack_regime_blocks': 0,
    'cooldown_activations': 0,
    'total_trades': 0,
    'sized_up_trades': 0,
    'daily_limit_blocks': 0,
    'session_scaled_down': 0
}

# =============================
# LOGGING SETUP
# =============================
def setup_logging():
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = "logs/backtests"
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/risk_engine_test_{date_str}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

logger = None

def get_session_multiplier(timestamp):
    """
    Apply Session Scaling
    Asia (00-08 UTC): 0.5x
    London (08-14 UTC): 1.0x
    NY (14-22 UTC): 1.0x
    Late (22-00 UTC): 0.5x
    """
    h = timestamp.hour
    if 0 <= h < 8: return 0.5, "Asia"
    if 22 <= h <= 23: return 0.5, "Late"
    return 1.0, "London/NY"

def run_backtest(args):
    global logger, stats
    
    # State flags
    ATTACK_MODE_ACTIVE = False
    losses_in_attack = 0
    attack_session_pnl = 0.0
    
    cooldown_counter = 0 
    
    # Daily PnL State
    current_date = None
    daily_pnl_r = 0.0
    daily_limit_hit = False
    
    # Validation Tracking
    trade_results = []
    
    # =============================
    # LOAD DATA
    # =============================
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH).copy()
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return

    df.sort_index(inplace=True)
    if 'fundingRate' not in df.columns: df['fundingRate'] = 0.0

    # =============================
    # INDICATORS
    # =============================
    df['range'] = df['high'] - df['low']
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range']
    df['lower_wick_pct'] = df['lower_wick'] / df['range']
    
    df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()
    
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
    df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()

    # =============================
    # SIMULATION LOOP
    # =============================
    balance = INITIAL_BALANCE
    active_bias = None
    last_cluster_end_idx = -999
    
    cooldown_until_idx = -1
    last_trade_pnl = 0
    recent_loss = False

    SWEEP_THINNING_MULT = 1.2
    SWEEP_WICK_PCT = 0.35
    MAX_BARS_SINCE_CLUSTER = 20
    ATR_TRAILING_STOP_MULT = 1.8
    INITIAL_STOP_ATR = 1.0
    COOLDOWN_ATR = 1.5

    print("Starting simulation with RISK ENGINE (Phase 2)...")
    
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        
        # -------------------------
        # DAILY PNL MANAGEMENT
        # -------------------------
        trade_date = timestamp.date()
        if current_date != trade_date:
            current_date = trade_date
            daily_pnl_r = 0.0
            daily_limit_hit = False
            # logger.info(f"New Day: {current_date}")
            
        if daily_limit_hit:
            stats['daily_limit_blocks'] += 1 # Rough count (per bar) - maybe misleading
            # Actually we should just skip logic if daily limit hit.
            continue 

        is_pocket = row['vol_pocket_active']
        
        # -------------------------
        # ATTACK MODE STATE
        # -------------------------
        kill_reason = None
        if ATTACK_MODE_ACTIVE:
            if losses_in_attack >= MAX_CONSECUTIVE_LOSSES:
                kill_reason = "Consecutive Losses"
                stats['attack_kills_loss'] += 1
            elif attack_session_pnl <= -MAX_DRAWDOWN_SESSION_R:
                kill_reason = "Session DD"
                stats['attack_kills_dd'] += 1
                
            if kill_reason:
                ATTACK_MODE_ACTIVE = False
                cooldown_counter = COOLDOWN_TRADES
                stats['cooldown_activations'] += 1

        if not ATTACK_MODE_ACTIVE:
            if cooldown_counter == 0:
                if (is_pocket and last_trade_pnl >= 0 and not recent_loss):
                    regime_ok = True
                    if len(trade_results) >= REGIME_WINDOW:
                        if np.mean(trade_results[-REGIME_WINDOW:]) < 0:
                            regime_ok = False
                            stats['attack_regime_blocks'] += 1
                    
                    if regime_ok:
                        ATTACK_MODE_ACTIVE = True
                        losses_in_attack = 0
                        attack_session_pnl = 0.0
                        stats['attack_activations'] += 1
        
        if i < cooldown_until_idx: continue

        # -------------------------
        # SIGNAL GENERATION
        # -------------------------
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        is_cluster = False
        if not (pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0):
            is_cluster = tr_cluster <= 0.7 * tr_prior

        if is_cluster:
            active_bias = None
            last_cluster_end_idx = i

        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        if is_thin_enough and (i - last_cluster_end_idx) <= MAX_BARS_SINCE_CLUSTER:
            prior_max = row['roll_max_20']
            prior_min = row['roll_min_20']
            
            is_valid_short = (row['upper_wick_pct'] > SWEEP_WICK_PCT and row['high'] > prior_max and row['close'] < prior_max)
            is_valid_long = (row['lower_wick_pct'] > SWEEP_WICK_PCT and row['low'] < prior_min and row['close'] > prior_min)

            if is_pocket:
                if is_valid_short: active_bias = 'SHORT'
                elif is_valid_long: active_bias = 'LONG'

        if active_bias is None: continue

        confirm = df.iloc[i + 1]
        failed = False
        if active_bias == 'SHORT': failed = confirm['close'] < row['close']
        else: failed = confirm['close'] > row['close']

        if not failed: continue
            
        # -------------------------
        # SIZING & RISK ENGINE SCALING
        # -------------------------
        
        base_size = 1.0
        
        # 1. Attack Mode Multiplier
        if ATTACK_MODE_ACTIVE:
            base_size = 1.5
            stats['sized_up_trades'] += 1
            
        # 2. Session Scaling (Risk Engine)
        session_mult = 1.0
        if ENABLE_SESSION_SCALING:
            session_mult, s_name = get_session_multiplier(timestamp)
            if session_mult < 1.0:
                stats['session_scaled_down'] += 1
        
        final_size = base_size * session_mult
            
        # Execution
        entry = confirm['close']
        atr = row['atr']
        
        if active_bias == 'LONG': stop = entry - INITIAL_STOP_ATR * atr
        else: stop = entry + INITIAL_STOP_ATR * atr

        exit_price = None
        R_raw = 0
        
        for j in range(i + 2, len(df)):
            current_bar = df.iloc[j]
            hi, lo, cl = current_bar['high'], current_bar['low'], current_bar['close']
            
            if active_bias == 'LONG':
                if lo <= stop:
                    exit_price = stop
                    R_raw = (exit_price - entry) / atr
                    break
                new_stop = cl - ATR_TRAILING_STOP_MULT * current_bar['atr']
                stop = max(stop, new_stop)
            else:
                if hi >= stop:
                    exit_price = stop
                    R_raw = (entry - exit_price) / atr
                    break
                new_stop = cl + ATR_TRAILING_STOP_MULT * current_bar['atr']
                stop = min(stop, new_stop)

        if exit_price is None: continue

        # Result calculation with Cost
        cost_r = (COST_BP / 10000) * 100 * final_size
        R_final = (R_raw * final_size) - cost_r
        
        balance += balance * RISK_PER_TRADE * R_final
        trade_results.append(R_final)
        stats['total_trades'] += 1
        
        # -------------------------
        # UPDATE DAILY PNL
        # -------------------------
        daily_pnl_r += R_final
        if daily_pnl_r <= DAILY_LOSS_LIMIT_R:
            daily_limit_hit = True
            # logger.warning(f"Daily Limit Hit @ {timestamp}: {daily_pnl_r:.2f}R")

        # -------------------------
        # UPDATE DAILY PNL
        # -------------------------
        daily_pnl_r += R_final
        if daily_pnl_r <= DAILY_LOSS_LIMIT_R:
            daily_limit_hit = True

        last_trade_pnl = R_final
        recent_loss = (R_final < 0)
        cooldown_until_idx = j + int(COOLDOWN_ATR * atr)
        active_bias = None
        
        # -------------------------
        # HARD STOP ANALYSIS (MAE)
        # -------------------------
        # Calculate Max Adverse Excursion during trade
        # df loc from i+2 to j
        if active_bias == 'LONG':
            min_price = df.iloc[i+2:j+1]['low'].min()
            mae_pct = (entry - min_price) / entry
        else:
            max_price = df.iloc[i+2:j+1]['high'].max()
            mae_pct = (max_price - entry) / entry
            
        # Check against theoretical Hard Stop (e.g. 5% and 10%)
        if mae_pct > 0.05:
            stats.setdefault('hard_stop_breach_5pct', 0)
            stats['hard_stop_breach_5pct'] += 1
        if mae_pct > 0.10:
            stats.setdefault('hard_stop_breach_10pct', 0)
            stats['hard_stop_breach_10pct'] += 1
            
        stats.setdefault('max_mae_observed', 0.0)
        if mae_pct > stats['max_mae_observed']:
             stats['max_mae_observed'] = mae_pct
        
        if ATTACK_MODE_ACTIVE:
            attack_session_pnl += R_final
            if R_final < 0: losses_in_attack += 1
            else: losses_in_attack = 0 
                
        if cooldown_counter > 0:
            cooldown_counter -= 1

    # =============================
    # RESULTS
    # =============================
    trades_arr = np.array(trade_results)
    
    peak = INITIAL_BALANCE
    current_bal = INITIAL_BALANCE
    max_dd_pct = 0.0
    
    for r in trade_results:
        pnl = current_bal * RISK_PER_TRADE * r
        current_bal += pnl
        if current_bal > peak: peak = current_bal
        dd = (peak - current_bal) / peak
        if dd > max_dd_pct: max_dd_pct = dd

    print("\n" + "="*40)
    print("RISK ENGINE BACKTEST RESULTS (Phase 2)")
    print("="*40)
    print(f"Final Balance:    {balance:.2f}")
    print(f"Max Drawdown:     {max_dd_pct*100:.2f}%")
    print(f"Total Trades:     {len(trades_arr)}")
    
    if len(trades_arr) > 0:
        print(f"Total R:          {trades_arr.sum():.2f}")
        print(f"Avg R:            {trades_arr.mean():.4f}")
    
    print("\nRisk Engine Metrics:")
    print(f"Session Scaled Down: {stats['session_scaled_down']}")
    print(f"Daily Limit Hit (Active Bars): {stats['daily_limit_blocks']}")
    print(f"Max MAE Observed: {stats.get('max_mae_observed', 0.0)*100:.2f}%")
    print(f"Hard Stop Breaches (5%): {stats.get('hard_stop_breach_5pct', 0)}")
    print(f"Hard Stop Breaches (10%): {stats.get('hard_stop_breach_10pct', 0)}")
    print("="*40)

def main():
    logger = setup_logging()
    run_backtest(None)

if __name__ == "__main__":
    main()
