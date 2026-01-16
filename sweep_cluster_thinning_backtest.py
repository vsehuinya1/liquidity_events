# sweep_cluster_thinning_backtest.py

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# =============================
# GLOBAL STATE & CONFIG
# =============================

DATA_PATH = 'data/parquet/SOLUSDT_1m_with_funding.parquet'
INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5  # round-trip
ATR_PERIOD = 20

# Attack Mode State
ATTACK_MODE_ALLOWED = False
ATTACK_MODE_ACTIVE = False
ATTACK_MODE_LOCKOUT = False

# Statistics
stats = {
    'attack_activations': 0,
    'attack_deactivations': 0,
    'attack_trades': 0,
    'addon_trades': 0,
    'funding_blocked': 0,
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
    log_file = f"{log_dir}/attack_mode_test_{date_str}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

logger = None  # Will be initialized in main

# =============================
# ATTACK MODE LOGIC
# =============================

def evaluate_attack_mode(context):
    """
    Evaluates and updates Attack Mode state based on context.
    Context keys: 'vol_pocket', 'last_trade_pnl', 'recent_loss', 'losses_in_attack'
    """
    global ATTACK_MODE_ALLOWED, ATTACK_MODE_ACTIVE, ATTACK_MODE_LOCKOUT, stats

    if not ATTACK_MODE_ALLOWED:
        return

    if ATTACK_MODE_LOCKOUT:
        ATTACK_MODE_ACTIVE = False
        return

    # Deactivation Logic
    if ATTACK_MODE_ACTIVE:
        reason = None
        if context.get('losses_in_attack', 0) >= 2:
            reason = "2 losses in attack mode"
            ATTACK_MODE_LOCKOUT = True
        elif not context['vol_pocket']:
            reason = "pocket invalid"
        # Note: Daily drawdown breach not implemented in this simplified backtest context
        
        if reason:
            ATTACK_MODE_ACTIVE = False
            stats['attack_deactivations'] += 1
            if logger: logger.info(f"[ATTACK] OFF @ {context['timestamp']}, reason: {reason}")
            return

    # Activation Logic
    # Mirroring live: pocket active, positive momentum (last trade winner), no recent loss
    if not ATTACK_MODE_ACTIVE:
        if (context['vol_pocket'] 
            and context.get('last_trade_pnl', 0) >= 0 
            and not context.get('recent_loss', False)):
            
            ATTACK_MODE_ACTIVE = True
            stats['attack_activations'] += 1
            if logger: logger.info(f"[ATTACK] ON @ {context['timestamp']}, Vol Pocket Active")

def run_backtest(args):
    global logger, ATTACK_MODE_ALLOWED, ATTACK_MODE_ACTIVE, ATTACK_MODE_LOCKOUT
    
    ATTACK_MODE_ALLOWED = args.attack_test
    
    # =============================
    # LOAD DATA
    # =============================
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH).copy()
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Run merge_data.py first.")
        return

    df.sort_index(inplace=True)
    
    # Ensure fundingRate exists (for funding filter)
    if 'fundingRate' not in df.columns:
        if args.funding_filter:
            print("Warning: fundingRate column missing. Funding filter will be ignored.")
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
    
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range']
    df['lower_wick_pct'] = df['lower_wick'] / df['range']
    
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
    # STATE INITIALIZATION
    # =============================
    balance = INITIAL_BALANCE
    trade_results = [] # Stores R multiple of each trade
    
    active_bias = None
    last_cluster_end_idx = -999
    prev_pocket_active = False
    
    cooldown_until = -1
    cluster_id = 0
    
    # Context State for Attack Mode
    last_trade_pnl = 0
    recent_loss = False
    losses_in_attack = 0

    # Locked Parameters
    SWEEP_THINNING_MULT = 1.2
    SWEEP_WICK_PCT = 0.35
    CLUSTER_LEN = 5
    COOLDOWN_ATR = 1.5
    MAX_BARS_SINCE_CLUSTER = 20
    ATR_TRAILING_STOP_MULT = 1.8
    INITIAL_STOP_ATR = 1.0

    print("Starting simulation...")
    
    # =============================
    # BACKTEST LOOP
    # =============================
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        
        # Volatility Pocket Update
        is_pocket = row['vol_pocket_active']
        
        # Evaluate Attack Mode State (at start of bar processing)
        context = {
            'timestamp': timestamp,
            'vol_pocket': is_pocket,
            'last_trade_pnl': last_trade_pnl,
            'recent_loss': recent_loss,
            'losses_in_attack': losses_in_attack,
            'funding_rate': row['fundingRate']
        }
        evaluate_attack_mode(context)
        
        if i < cooldown_until:
            continue

        # =========================
        # CLUSTER DETECTION (COMPRESSION)
        # =========================
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
            is_cluster = False
        else:
            is_cluster = tr_cluster <= 0.7 * tr_prior

        if is_cluster:
            cluster_id += 1
            active_bias = None
            last_cluster_end_idx = i

        # =========================
        # SWEEP DETECTION
        # =========================
        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        
        if is_thin_enough:
            if (i - last_cluster_end_idx) > MAX_BARS_SINCE_CLUSTER:
                active_bias = None
            else:
                prior_max = row['roll_max_20']
                prior_min = row['roll_min_20']
                
                is_valid_short = (
                    row['upper_wick_pct'] > SWEEP_WICK_PCT and 
                    row['high'] > prior_max and
                    row['close'] < prior_max
                )
                
                is_valid_long = (
                    row['lower_wick_pct'] > SWEEP_WICK_PCT and
                    row['low'] < prior_min and
                    row['close'] > prior_min
                )

                if is_pocket:
                    if is_valid_short:
                        active_bias = 'SHORT'
                    elif is_valid_long:
                        active_bias = 'LONG'

        # =========================
        # ENTRY — FAILED CONTINUATION
        # =========================
        if active_bias is None:
            continue

        confirm = df.iloc[i + 1]

        if active_bias == 'SHORT':
            failed = confirm['close'] < row['close']
        else:
            failed = confirm['close'] > row['close']

        if not failed:
            continue
            
        # =========================
        # FUNDING FILTER
        # =========================
        if args.funding_filter:
            funding = row['fundingRate']
            blocked = False
            # Funding rate is periodic rate.
            # Longs pay shorts if funding > 0.
            # High positive funding -> avoid longs (or expect reversion).
            if active_bias == 'LONG' and funding > args.funding_pos_th:
                blocked = True
                if logger: logger.info(f"[FUNDING] Blocked LONG {timestamp}, Rate: {funding:.6f} > {args.funding_pos_th}")
            elif active_bias == 'SHORT' and funding < -args.funding_neg_th:
                blocked = True
                if logger: logger.info(f"[FUNDING] Blocked SHORT {timestamp}, Rate: {funding:.6f} < -{args.funding_neg_th}")
            
            if blocked:
                stats['funding_blocked'] += 1
                active_bias = None
                continue

        # =========================
        # EXECUTE TRADE
        # =========================
        entry = confirm['close']
        atr = row['atr']
        
        # Initial Stop
        if active_bias == 'LONG':
            stop = entry - INITIAL_STOP_ATR * atr
        else:
            stop = entry + INITIAL_STOP_ATR * atr

        exit_price = None
        R = 0
        addon_taken = False
        addon_entry = None
        
        trade_start_idx = i + 2
        
        for j in range(trade_start_idx, len(df)):
            current_bar = df.iloc[j]
            hi = current_bar['high']
            lo = current_bar['low']
            cl = current_bar['close']
            
            # -------------------------
            # ADD-ON LOGIC
            # -------------------------
            if (ATTACK_MODE_ACTIVE and args.addon_test and not addon_taken):
                # Simple check: Is current unadjusted PnL > 0?
                # This is a simplification. Real add-on logic might look for specific pullbacks.
                # Assuming "if trade is working" means price is favorable.
                is_profitable = False
                if active_bias == 'LONG' and cl > entry: is_profitable = True
                elif active_bias == 'SHORT' and cl < entry: is_profitable = True
                
                if is_profitable:
                    addon_size = 0.5 # relative to initial size
                    addon_entry = cl # Enter at close of check bar
                    addon_taken = True
                    stats['addon_trades'] += 1
                    if logger: logger.info(f"[ADDON] Triggered for {timestamp} trade at {addon_entry}")

            # -------------------------
            # TRAILING STOP EXIT
            # -------------------------
            if active_bias == 'LONG':
                # Check Breach
                if lo <= stop:
                    exit_price = stop
                    # Calculate R
                    # Initial Position: size 1.0, risk 1 ATR approx.
                    initial_R = (exit_price - entry) / atr
                    
                    # Add-on Position: size 0.5
                    addon_R = 0
                    if addon_taken:
                        addon_R = 0.5 * ((exit_price - addon_entry) / atr)
                        
                    R = initial_R + addon_R
                    break
                
                # Update Trail
                new_stop = cl - ATR_TRAILING_STOP_MULT * current_bar['atr']
                stop = max(stop, new_stop)
                
            else: # SHORT
                # Check Breach
                if hi >= stop:
                    exit_price = stop
                    initial_R = (entry - exit_price) / atr
                    
                    addon_R = 0
                    if addon_taken:
                        addon_R = 0.5 * ((addon_entry - exit_price) / atr)
                        
                    R = initial_R + addon_R
                    break
                    
                # Update Trail
                new_stop = cl + ATR_TRAILING_STOP_MULT * current_bar['atr']
                stop = min(stop, new_stop)

        if exit_price is None:
            continue

        cost_mult = 1.5 if addon_taken else 1.0
        # Cost approx in R. 1.5bps = 0.015% price. If 1 ATR risk is approx 1%, then 0.015% is 0.015R
        # Let's approximate R cost per trade as:
        # R cost = (COST_BP / 10000) * (EntryPrice / RiskAmt)
        # Typically RiskAmt ~ 1% of Price. So Price/Risk ~ 100.
        # R Cost ~ 0.00015 * 100 = 0.015 R.
        # We will use simplified 0.02 R cost for baseline.
        R -= 0.02 * cost_mult 
        
        balance += balance * RISK_PER_TRADE * R
        trade_results.append(R)
        stats['total_trades'] += 1
        
        if ATTACK_MODE_ACTIVE:
            stats['attack_trades'] += 1
            if R < 0:
                losses_in_attack += 1
            else:
                pass # Losses persist until reset or lockout? "2 losses in attack mode" -> cumulative
        
        last_trade_pnl = R
        recent_loss = (R < 0)
        
        if is_pocket:
            pocket_traded = True
            
        cooldown_until = j + int(COOLDOWN_ATR * atr)
        active_bias = None

    # =============================
    # FINAL REPORTING
    # =============================
    trades_arr = np.array(trade_results)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS")
    print("="*40)
    print(f"Final Balance:    {balance:.2f}")
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
    print(f"Attack Mode Active: {stats['attack_activations']} times")
    print(f"Attack Mode Trades: {stats['attack_trades']}")
    print(f"Add-on Trades:      {stats['addon_trades']}")
    print(f"Funding Blocked:    {stats['funding_blocked']}")
    print("="*40)

def main():
    global logger
    
    parser = argparse.ArgumentParser(description='Sweep Backtest with Attack Mode')
    parser.add_argument('--attack-test', action='store_true', help='Enable Attack Mode logic')
    parser.add_argument('--funding-filter', action='store_true', help='Enable Funding Frequency logic')
    parser.add_argument('--addon-test', action='store_true', help='Enable Add-on logic')
    
    parser.add_argument('--funding-pos-th', type=float, default=0.01, help='Positive funding threshold (default 0.01)')
    parser.add_argument('--funding-neg-th', type=float, default=0.01, help='Negative funding threshold (default 0.01)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    print(f"Running Backtest with:")
    print(f"Attack Mode:    {'ON' if args.attack_test else 'OFF'}")
    print(f"Funding Filter: {'ON' if args.funding_filter else 'OFF'}")
    print(f"Add-ons:        {'ON' if args.addon_test else 'OFF'}")
    
    run_backtest(args)

if __name__ == "__main__":
    main()
