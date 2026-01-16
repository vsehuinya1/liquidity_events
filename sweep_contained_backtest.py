# sweep_contained_backtest.py

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

# Containment Config
MAX_DRAWDOWN_SESSION_R = 2.0  # Kill Attack Mode if session drawdown > 2R
MAX_CONSECUTIVE_LOSSES = 2    # Kill Attack Mode if >= 2 losses
COOLDOWN_TRADES = 5           # Trades to sit out aggressive mode after kill
REGIME_WINDOW = 10            # Rolling window for regime filter

# Statistics
stats = {
    'attack_activations': 0,
    'attack_kills_dd': 0,
    'attack_kills_loss': 0,
    'attack_regime_blocks': 0,
    'cooldown_activations': 0,
    'total_trades': 0,
    'sized_up_trades': 0
}

# =============================
# LOGGING SETUP
# =============================
def setup_logging():
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = "logs/backtests"
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/contained_test_{date_str}.log"
    
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

def run_backtest(args):
    global logger, stats
    
    # State flags
    ATTACK_MODE_ACTIVE = False
    losses_in_attack = 0
    attack_session_pnl = 0.0
    
    cooldown_counter = 0 # Trades remaining in cooldown
    
    # Validation Tracking
    trade_results = [] # History of all R outcomes
    
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
    # Ensure columns exist
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
    
    # Wick %
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range']
    df['lower_wick_pct'] = df['lower_wick'] / df['range']
    
    # Sweep Lookback
    df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()
    
    # Volatility Pocket
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    # Cluster Helpers
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

    # Locked Strategy Parameters
    SWEEP_THINNING_MULT = 1.2
    SWEEP_WICK_PCT = 0.35
    MAX_BARS_SINCE_CLUSTER = 20
    ATR_TRAILING_STOP_MULT = 1.8
    INITIAL_STOP_ATR = 1.0
    COOLDOWN_ATR = 1.5

    print("Starting simulation with CONTAINMENT...")
    
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        is_pocket = row['vol_pocket_active']
        
        # -------------------------
        # ATTACK MODE STATE MANAGEMENT
        # -------------------------
        
        # 1. Damage-Based Invalidation (Check BEFORE activation logic)
        kill_reason = None
        if ATTACK_MODE_ACTIVE:
            # 1.1 Consecutive Loss Kill
            if losses_in_attack >= MAX_CONSECUTIVE_LOSSES:
                kill_reason = f"{MAX_CONSECUTIVE_LOSSES} consecutive losses"
                stats['attack_kills_loss'] += 1
                
            # 1.2 Session Drawdown Kill
            # Note: We track session_pnl cumulatively.
            elif attack_session_pnl <= -MAX_DRAWDOWN_SESSION_R:
                kill_reason = f"Session DD {attack_session_pnl:.2f}R hit cap"
                stats['attack_kills_dd'] += 1
                
            if kill_reason:
                ATTACK_MODE_ACTIVE = False
                cooldown_counter = COOLDOWN_TRADES # Start cooldown
                stats['cooldown_activations'] += 1
                if logger: logger.info(f"[ATTACK_KILL] OFF @ {timestamp}, Reason: {kill_reason}")

        # 2. Activation Logic
        if not ATTACK_MODE_ACTIVE:
            # Check Cooldown
            if cooldown_counter > 0:
                # Still in cooldown, cannot activate
                pass
            else:
                # Normal Activation Check
                if (is_pocket and last_trade_pnl >= 0 and not recent_loss):
                    # 2.1 Regime Governor Check (Rolling Expectancy)
                    regime_ok = True
                    if len(trade_results) >= REGIME_WINDOW:
                        rolling_R = np.mean(trade_results[-REGIME_WINDOW:])
                        if rolling_R < 0:
                            regime_ok = False
                            stats['attack_regime_blocks'] += 1
                            # Fails silently? Or log?
                            # if logger: logger.info(f"[REGIME_BLOCK] Expectancy {rolling_R:.2f} < 0")
                    
                    if regime_ok:
                        ATTACK_MODE_ACTIVE = True
                        losses_in_attack = 0
                        attack_session_pnl = 0.0
                        stats['attack_activations'] += 1
                        if logger: logger.info(f"[ATTACK] ON @ {timestamp}, Regime OK")
        
        if i < cooldown_until_idx:
            continue

        # -------------------------
        # SIGNAL GENERATION
        # -------------------------
        
        # Cluster
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
            is_cluster = False
        else:
            is_cluster = tr_cluster <= 0.7 * tr_prior

        if is_cluster:
            active_bias = None
            last_cluster_end_idx = i

        # Sweep
        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        if is_thin_enough and (i - last_cluster_end_idx) <= MAX_BARS_SINCE_CLUSTER:
            prior_max = row['roll_max_20']
            prior_min = row['roll_min_20']
            
            is_valid_short = (row['upper_wick_pct'] > SWEEP_WICK_PCT and row['high'] > prior_max and row['close'] < prior_max)
            is_valid_long = (row['lower_wick_pct'] > SWEEP_WICK_PCT and row['low'] < prior_min and row['close'] > prior_min)

            if is_pocket:
                if is_valid_short: active_bias = 'SHORT'
                elif is_valid_long: active_bias = 'LONG'

        # Entry Confirmation
        if active_bias is None:
            continue

        confirm = df.iloc[i + 1]
        if active_bias == 'SHORT': failed = confirm['close'] < row['close']
        else: failed = confirm['close'] > row['close']

        if not failed:
            continue
            
        # -------------------------
        # SIZING & EXECUTION
        # -------------------------
        
        position_size_mult = 1.0
        
        # Aggressive Sizing (If Active & Not Cooldown)
        # Note: If cooldown > 0, ATTACK_MODE_ACTIVE is forced false above roughly.
        if ATTACK_MODE_ACTIVE:
            position_size_mult = 1.5
            stats['sized_up_trades'] += 1
            
        # Execute Trade (Simulation)
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

        if exit_price is None:
            continue

        # Adjust for Size and Cost
        cost_r = (COST_BP / 10000) * 100 * position_size_mult
        R_final = (R_raw * position_size_mult) - cost_r
        
        balance += balance * RISK_PER_TRADE * R_final
        trade_results.append(R_final)
        stats['total_trades'] += 1
        
        # -------------------------
        # POST-TRADE STATE UPDATE
        # -------------------------
        
        last_trade_pnl = R_final
        recent_loss = (R_final < 0)
        cooldown_until_idx = j + int(COOLDOWN_ATR * atr)
        active_bias = None
        
        # Update Containment Counters
        if ATTACK_MODE_ACTIVE:
            attack_session_pnl += R_final
            if R_final < 0:
                losses_in_attack += 1
            else:
                # Do we reset losses on win?
                # "Track consecutive losing trades" -> implies reset on win.
                losses_in_attack = 0 
                
        # Update Cooldown Counter (Decrements on every trade taken)
        if cooldown_counter > 0:
            cooldown_counter -= 1
            if cooldown_counter == 0:
                if logger: logger.info(f"[COOLDOWN] END @ {timestamp}")

    # =============================
    # RESULTS REPORT
    # =============================
    trades_arr = np.array(trade_results)
    
    # Calculate Max Drawdown
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
    print("CONTAINED BACKTEST RESULTS")
    print("="*40)
    print(f"Final Balance:    {balance:.2f}")
    print(f"Max Drawdown:     {max_dd_pct*100:.2f}%")
    print(f"Total Trades:     {len(trades_arr)}")
    
    if len(trades_arr) > 0:
        print(f"Total R:          {trades_arr.sum():.2f}")
        print(f"Avg R:            {trades_arr.mean():.4f}")
    
    print("\nContainment Metrics:")
    print(f"Attack Activations: {stats['attack_activations']}")
    print(f"Kills (Losses):     {stats['attack_kills_loss']}")
    print(f"Kills (Drawdown):   {stats['attack_kills_dd']}")
    print(f"Regime Blocks:      {stats['attack_regime_blocks']}")
    print(f"Cooldowns:          {stats['cooldown_activations']}")
    print(f"Sized Up Trades:    {stats['sized_up_trades']}")
    print("="*40)

def main():
    logger = setup_logging()
    run_backtest(None)

if __name__ == "__main__":
    main()
