# sweep_optimized_backtest.py

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

# Statistics
stats = {
    'attack_activations': 0,
    'attack_deactivations': 0,
    'attack_trades': 0,
    'addon_trades': 0, # Used for aggressive entry count
    'funding_squeeze_trades': 0,
    'total_trades': 0
}

# =============================
# LOGGING SETUP
# =============================
def setup_logging():
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = "logs/backtests"
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/optimized_test_{date_str}.log"
    
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
    if 'fundingRate' not in df.columns:
        df['fundingRate'] = 0.0

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
    
    cooldown_until = -1
    last_trade_pnl = 0
    recent_loss = False

    # Locked Parameters
    SWEEP_THINNING_MULT = 1.2
    SWEEP_WICK_PCT = 0.35
    MAX_BARS_SINCE_CLUSTER = 20
    ATR_TRAILING_STOP_MULT = 1.8
    INITIAL_STOP_ATR = 1.0
    COOLDOWN_ATR = 1.5

    print("Starting simulation...")
    
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        is_pocket = row['vol_pocket_active']
        
        # -------------------------
        # ATTACK MODE LOGIC (STICKY)
        # -------------------------
        if args.sticky_attack:
            if ATTACK_MODE_ACTIVE:
                # Deactivate ONLY on cumulative losses.
                # Pocket status is IGNORED for deactivation in Sticky mode.
                if losses_in_attack >= 2:
                    ATTACK_MODE_ACTIVE = False
                    stats['attack_deactivations'] += 1
                    if logger: logger.info(f"[ATTACK] OFF @ {timestamp}, reason: 2 losses")
            else:
                # Activate on: Pocket + Momentum
                if (is_pocket and last_trade_pnl >= 0 and not recent_loss):
                    ATTACK_MODE_ACTIVE = True
                    losses_in_attack = 0 # Reset counter
                    stats['attack_activations'] += 1
                    if logger: logger.info(f"[ATTACK] ON @ {timestamp}, Vol Pocket Active")
        
        if i < cooldown_until:
            continue

        # -------------------------
        # CLUSTER DETECTION
        # -------------------------
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
            is_cluster = False
        else:
            is_cluster = tr_cluster <= 0.7 * tr_prior

        if is_cluster:
            active_bias = None
            last_cluster_end_idx = i

        # -------------------------
        # SWEEP DETECTION
        # -------------------------
        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        
        if is_thin_enough and (i - last_cluster_end_idx) <= MAX_BARS_SINCE_CLUSTER:
            prior_max = row['roll_max_20']
            prior_min = row['roll_min_20']
            
            is_valid_short = (row['upper_wick_pct'] > SWEEP_WICK_PCT and row['high'] > prior_max and row['close'] < prior_max)
            is_valid_long = (row['lower_wick_pct'] > SWEEP_WICK_PCT and row['low'] < prior_min and row['close'] > prior_min)

            if is_pocket:
                if is_valid_short: active_bias = 'SHORT'
                elif is_valid_long: active_bias = 'LONG'

        # -------------------------
        # ENTRY
        # -------------------------
        if active_bias is None:
            continue

        confirm = df.iloc[i + 1]
        
        # Failed Continuation Check
        if active_bias == 'SHORT': failed = confirm['close'] < row['close']
        else: failed = confirm['close'] > row['close']

        if not failed:
            continue
            
        # -------------------------
        # POSITION SIZING LOGIC
        # -------------------------
        position_size_mult = 1.0
        
        # EXPERIMENT B: FUNDING SQUEEZE
        # If signal is against funding flow (Contracting crowd), size up.
        # Long Signal + Neg Funding (Shorts Paying Longs) -> Short Squeeze Potential
        # Short Signal + Pos Funding (Longs Paying Shorts) -> Long Squeeze Potential
        funding = row['fundingRate']
        is_squeeze = False
        
        if args.funding_squeeze:
            if active_bias == 'LONG' and funding < -0.0001: # Significant neg funding
                is_squeeze = True
                if logger: logger.info(f"[SQUEEZE] LONG setup with Neg Funding {funding:.5f}")
            elif active_bias == 'SHORT' and funding > 0.0001: # Significant pos funding
                is_squeeze = True
                if logger: logger.info(f"[SQUEEZE] SHORT setup with Pos Funding {funding:.5f}")
            
            if is_squeeze:
                position_size_mult += 0.25 # +25% size
                stats['funding_squeeze_trades'] += 1

        # EXPERIMENT C: AGGRESSIVE ADD-ON (UPFRONT)
        # If Attack Mode active, size up immediately.
        if args.aggressive_addon and ATTACK_MODE_ACTIVE:
            position_size_mult += 0.5 # +50% size
            stats['addon_trades'] += 1
            if logger: logger.info(f"[AGGRESSIVE] Upfront size increase. Total Mult: {position_size_mult}")

        # -------------------------
        # EXECUTE TRADE
        # -------------------------
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

        # Adjust R for Position Size
        # R_final = R_raw * Size_Mult
        # Cost is also proportional to size
        cost_r = (COST_BP / 10000) * 100 * position_size_mult # Approx 0.015 * Mult
        R_final = (R_raw * position_size_mult) - cost_r
        
        balance += balance * RISK_PER_TRADE * R_final
        trade_results.append(R_final)
        stats['total_trades'] += 1
        
        # Update State
        last_trade_pnl = R_final
        recent_loss = (R_final < 0)
        
        if ATTACK_MODE_ACTIVE:
            stats['attack_trades'] += 1
            if R_final < 0:
                losses_in_attack += 1
        
        cooldown_until = j + int(COOLDOWN_ATR * atr)
        active_bias = None

    # =============================
    # RESULTS REPORT
    # =============================
    trades_arr = np.array(trade_results)
    
    # Calculate Max Drawdown
    # Reconstruct equity curve
    equity = [INITIAL_BALANCE]
    current_bal = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd_pct = 0.0
    
    for r in trade_results:
        # Re-apply cost/risk logic to get exact dollar amt if needed, 
        # but we already have 'balance' updated in loop? 
        # Actually 'trade_results' stores R. We need dollar PnL sequence or just use final balance?
        # The loop updated 'balance' cumulatively. We didn't store equity curve points.
        # Let's reconstruct roughly or just track it in the loop. 
        # Better: Reuse the loop logic or approximate from R if fixed risk.
        # Since we have compound growth (balance += ...), let's re-run calc on the R array.
        
        pnl = current_bal * RISK_PER_TRADE * r
        current_bal += pnl
        equity.append(current_bal)
        
        if current_bal > peak:
            peak = current_bal
        
        dd = (peak - current_bal) / peak
        if dd > max_dd_pct:
            max_dd_pct = dd

    print("\n" + "="*40)
    print("OPTIMIZED BACKTEST RESULTS")
    print("="*40)
    print(f"Final Balance:    {balance:.2f}")
    print(f"Max Drawdown:     {max_dd_pct*100:.2f}%")
    print(f"Total Trades:     {len(trades_arr)}")
    
    if len(trades_arr) > 0:
        win_rate = (trades_arr > 0).mean()
        avg_r = trades_arr.mean()
        print(f"Win Rate:         {win_rate:.4f}")
        print(f"Avg R:            {avg_r:.4f}")
        print(f"Total R:          {trades_arr.sum():.2f}")
    else:
        print("No trades taken.")

    print("\nFeature Metrics:")
    print(f"Attack Mode Active:     {stats['attack_activations']} times")
    print(f"Attack Trades:          {stats['attack_trades']}")
    print(f"Aggressive Add-ons:     {stats['addon_trades']}")
    print(f"Funding Squeeze Trades: {stats['funding_squeeze_trades']}")
    print("="*40)

def main():
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--sticky-attack', action='store_true', help='Experiment A: Sticky Attack Mode')
    parser.add_argument('--funding-squeeze', action='store_true', help='Experiment B: Funding Squeeze Sizing')
    parser.add_argument('--aggressive-addon', action='store_true', help='Experiment C: Aggressive Upfront Addon')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    print(f"Running Optimized Backtest with:")
    print(f"Sticky Attack:   {'ON' if args.sticky_attack else 'OFF'}")
    print(f"Funding Squeeze: {'ON' if args.funding_squeeze else 'OFF'}")
    print(f"Aggressive Size: {'ON' if args.aggressive_addon else 'OFF'}")
    
    run_backtest(args)

if __name__ == "__main__":
    main()
