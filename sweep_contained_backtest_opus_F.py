# sweep_contained_backtest_opus_F.py
# OPUS OPTION F: FUSION (DYNAMIC FILTERING)
# Base Mode: Strict Filters (Survival)
# Burst Mode: Loose Filters (Aggression)

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# =============================
# CONFIGURATION
# =============================

INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5
ATR_PERIOD = 20

# -----------------------------
# MODE A: BURST (AGGRESSOR)
# -----------------------------
BURST_CONFIG = {
    'MAX_DRAWDOWN_SESSION_R': 10.0,
    'MAX_CONSECUTIVE_LOSSES': 10,
    'COOLDOWN_TRADES': 0,
    'MIN_REGIME_EXPECTANCY': -5.0,
    'MAX_DRAWDOWN_SESSION_R': 10.0,
    'MAX_CONSECUTIVE_LOSSES': 10,
    'COOLDOWN_TRADES': 0,
    'MIN_REGIME_EXPECTANCY': -5.0,
    'MAX_DRAWDOWN_SESSION_R': 10.0,
    'MAX_CONSECUTIVE_LOSSES': 10,
    'COOLDOWN_TRADES': 0,
    'MIN_REGIME_EXPECTANCY': -5.0,
    'ATTACK_SIZE_MULT': 2.0,  # Reduced from 2.2
    'HOT_STREAK_SIZE_MULT': 2.5, # Reduced from 3.0
    'FUNDING_SQUEEZE_BONUS': 0.5
}

# -----------------------------
# MODE C: BASE (SURVIVAL)
# -----------------------------
BASE_CONFIG = {
    'MAX_DRAWDOWN_SESSION_R': 2.0,  # Keep 2.0
    'MAX_CONSECUTIVE_LOSSES': 3,    # Strict limit
    'COOLDOWN_TRADES': 8,           # Increased from 6
    'MIN_REGIME_EXPECTANCY': -0.3,
    'ATTACK_SIZE_MULT': 1.5,
    'HOT_STREAK_SIZE_MULT': 1.8,
    'FUNDING_SQUEEZE_BONUS': 0.2
}

# FILTER CONFIGS (DYNAMIC)
# -----------------------------
# MODE 1: BASE (SAFETY) - Strict Filters
BASE_FILTERS = {
    'SWEEP_WICK_PCT': 0.45,
    'CLUSTER_COMPRESSION_RATIO': 0.6
}

# MODE 2: BURST (ATTACK) - Loose Filters
BURST_FILTERS = {
    'SWEEP_WICK_PCT': 0.35,      # Original Loose
    'CLUSTER_COMPRESSION_RATIO': 0.7 # Original Loose
}

# Shared Params
SWEEP_THINNING_MULT = 1.2      
MAX_BARS_SINCE_CLUSTER = 20    
ATR_TRAILING_STOP_MULT = 1.8
INITIAL_STOP_ATR = 0.95 
COOLDOWN_ATR = 1.2

# Statistics
stats = {
    'attack_activations': 0,
    'burst_mode_activations': 0, # NEW
    'burst_mode_trades': 0,      # NEW
    'burst_mode_kills': 0,       # NEW
    'attack_kills_dd': 0,
    'attack_kills_loss': 0,
    'attack_regime_blocks': 0,
    'cooldown_activations': 0,
    'total_trades': 0,
    'sized_up_trades': 0,
    'hot_streak_trades': 0,
    'funding_aligned_trades': 0
}

# =============================
# LOGGING SETUP
# =============================
def setup_logging():
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = "logs/backtests"
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/opus_D_test_{date_str}.log"
    
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

def get_current_config(burst_active):
    return BURST_CONFIG if burst_active else BASE_CONFIG

def run_backtest(args):
    global logger, stats
    
    data_path = args.data_path if args and args.data_path else 'data/parquet/SOLUSDT_1m_with_funding.parquet'
    
    # State flags
    ATTACK_MODE_ACTIVE = False
    BURST_MODE_ACTIVE = False # NEW
    
    losses_in_attack = 0
    wins_in_attack = 0
    attack_session_pnl = 0.0
    
    cooldown_counter = 0
    
    trade_results = []
    trade_log = [] # New list for CSV
    consecutive_wins = 0 # Global win streak tracker
    
    # =============================
    # LOAD DATA
    # =============================
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path).copy()
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
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
    df['upper_wick_pct'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['range'].replace(0, np.nan)
    
    df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()
    
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
    df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()

    # Pre-calculate Filters NOT possible globally anymore (Dynamic)
    # But we can pre-calculate WICK metrics
    # Wick PCT is already in DF. Logic is handled in loop.

    # =============================
    # SIMULATION LOOP
    # =============================
    balance = INITIAL_BALANCE
    active_bias = None
    last_cluster_end_idx = -999
    
    cooldown_until_idx = -1
    last_trade_pnl = 0
    recent_loss = False

    print("Starting OPUS D (Dynamic) simulation...")
    
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        is_pocket = row['vol_pocket_active']
        
        # -------------------------
        # BURST MODE LOGIC (NEW)
        # -------------------------
        # Trigger: Rolling R > 0.5 (Last 10) OR Win Streak >= 2
        # Disable: Any Loss OR Rolling R Drops
        
        rolling_R = 0.0
        if len(trade_results) >= 10:
            rolling_R = np.mean(trade_results[-10:])
        
        if not BURST_MODE_ACTIVE:
            # Check triggers - REQUIRE NO RECENT LOSS
            if not recent_loss and (consecutive_wins >= 3 or rolling_R > 0.7):
                BURST_MODE_ACTIVE = True
                stats['burst_mode_activations'] += 1
                if logger: logger.info(f"[BURST] ON @ {timestamp} | Streak: {consecutive_wins} | Roll R: {rolling_R:.2f}")
        else:
            # Check disable triggers
            # 1. Any Loss (Immediate Cutoff)
            if recent_loss: 
                BURST_MODE_ACTIVE = False
                stats['burst_mode_kills'] += 1
                if logger: logger.info(f"[BURST] OFF @ {timestamp} | Loss Detected")
            
            # 2. Edge Degradation
            elif len(trade_results) >= 10 and rolling_R < 0.2:
                BURST_MODE_ACTIVE = False
                if logger: logger.info(f"[BURST] OFF @ {timestamp} | Edge Faded (R={rolling_R:.2f})")

        # Get Config based on State
        config = get_current_config(BURST_MODE_ACTIVE)
        
        # Get Current Filters
        current_filters = BURST_FILTERS if BURST_MODE_ACTIVE else BASE_FILTERS
        current_wick_pct = current_filters['SWEEP_WICK_PCT']
        current_compression = current_filters['CLUSTER_COMPRESSION_RATIO']
        
        # -------------------------
        # ATTACK MODE STATE MANAGEMENT
        # -------------------------
        
        # 1. Damage-Based Invalidation
        kill_reason = None
        if ATTACK_MODE_ACTIVE:
            if losses_in_attack >= config['MAX_CONSECUTIVE_LOSSES']:
                kill_reason = f"{config['MAX_CONSECUTIVE_LOSSES']} consecutive losses"
                stats['attack_kills_loss'] += 1
                
            elif attack_session_pnl <= -config['MAX_DRAWDOWN_SESSION_R']:
                kill_reason = f"Session DD {attack_session_pnl:.2f}R hit cap"
                stats['attack_kills_dd'] += 1
                
            if kill_reason:
                ATTACK_MODE_ACTIVE = False
                cooldown_counter = config['COOLDOWN_TRADES']
                stats['cooldown_activations'] += 1
                wins_in_attack = 0
                
                # Force burst off on kill
                if BURST_MODE_ACTIVE:
                    BURST_MODE_ACTIVE = False
                    stats['burst_mode_kills'] += 1

        # 2. Activation Logic
        if not ATTACK_MODE_ACTIVE:
            if cooldown_counter > 0:
                pass
            else:
                activation_condition = (
                    is_pocket 
                    and last_trade_pnl >= 0 
                    and not recent_loss
                )
                
                if activation_condition:
                    regime_ok = True
                    if len(trade_results) >= 12: # Check rolling window
                        # Use Config window? Or fixed? Let's use fixed 10 for consistency
                        current_rolling = np.mean(trade_results[-10:])
                        if current_rolling < config['MIN_REGIME_EXPECTANCY']:
                            regime_ok = False
                            stats['attack_regime_blocks'] += 1
                    
                    if regime_ok:
                        ATTACK_MODE_ACTIVE = True
                        losses_in_attack = 0
                        wins_in_attack = 0
                        attack_session_pnl = 0.0
                        stats['attack_activations'] += 1
        
        if i < cooldown_until_idx:
            continue

        # -------------------------
        # SIGNAL GENERATION
        # -------------------------
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
            is_cluster = False
        else:
            is_cluster = tr_cluster <= current_compression * tr_prior # DYNAMIC COMPRESSION

        if is_cluster:
            active_bias = None
            last_cluster_end_idx = i

        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        bars_since_cluster = i - last_cluster_end_idx
        
        if is_thin_enough and bars_since_cluster <= MAX_BARS_SINCE_CLUSTER:
            prior_max = row['roll_max_20']
            prior_min = row['roll_min_20']
            
            is_valid_short = (
                row['upper_wick_pct'] > current_wick_pct # DYNAMIC WICK
                and row['high'] > prior_max 
                and row['close'] < prior_max
            )
            is_valid_long = (
                row['lower_wick_pct'] > current_wick_pct # DYNAMIC WICK
                and row['low'] < prior_min 
                and row['close'] > prior_min
            )
            
            if is_pocket:
                if is_valid_short: active_bias = 'SHORT'
                elif is_valid_long: active_bias = 'LONG'

        if active_bias is None:
            continue

        confirm = df.iloc[i + 1]
        if active_bias == 'SHORT': failed = confirm['close'] < row['close']
        else: failed = confirm['close'] > row['close']

        if not failed:
            continue
            
        # -------------------------
        # SIZING
        # -------------------------
        position_size_mult = 1.0
        
        if ATTACK_MODE_ACTIVE:
            position_size_mult = config['ATTACK_SIZE_MULT']
            stats['sized_up_trades'] += 1
            if BURST_MODE_ACTIVE:
               stats['burst_mode_trades'] += 1
            
            if wins_in_attack >= 2:
                position_size_mult = config['HOT_STREAK_SIZE_MULT']
                stats['hot_streak_trades'] += 1
        
        # Funding Bonus
        funding = row['fundingRate']
        funding_aligned = False
        if active_bias == 'LONG' and funding < -0.0001: funding_aligned = True
        elif active_bias == 'SHORT' and funding > 0.0001: funding_aligned = True
            
        if funding_aligned:
            position_size_mult += config['FUNDING_SQUEEZE_BONUS']
            stats['funding_aligned_trades'] += 1
            
        # -------------------------
        # EXECUTE
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
            current_atr = current_bar['atr']
            
            if active_bias == 'LONG':
                if lo <= stop:
                    exit_price = stop
                    R_raw = (exit_price - entry) / atr
                    break
                new_stop = cl - ATR_TRAILING_STOP_MULT * current_atr
                stop = max(stop, new_stop)
            else:
                if hi >= stop:
                    exit_price = stop
                    R_raw = (entry - exit_price) / atr
                    break
                new_stop = cl + ATR_TRAILING_STOP_MULT * current_atr
                stop = min(stop, new_stop)

        if exit_price is None:
            continue

        cost_r = (COST_BP / 10000) * 100 * position_size_mult
        R_final = (R_raw * position_size_mult) - cost_r
        
        balance += balance * RISK_PER_TRADE * R_final
        trade_results.append(R_final)
        stats['total_trades'] += 1
        
        # -------------------------
        # POST-TRADE UPDATE
        # -------------------------
        last_trade_pnl = R_final
        recent_loss = (R_final < 0)
        cooldown_until_idx = j + int(COOLDOWN_ATR * atr)
        active_bias = None
        
        if R_final > 0:
            consecutive_wins += 1
        else:
            consecutive_wins = 0
        
        if ATTACK_MODE_ACTIVE:
            attack_session_pnl += R_final
            if R_final < 0:
                losses_in_attack += 1
                wins_in_attack = 0
            else:
                losses_in_attack = 0
                wins_in_attack += 1
                
        if cooldown_counter > 0:
            cooldown_counter -= 1
            
        # Log trade for CSV
        trade_log.append({
            'timestamp': timestamp,
            'bias': active_bias,
            'entry_price': entry,
            'exit_price': exit_price,
            'R_raw': R_raw,
            'size_mult': position_size_mult,
            'R_final': R_final,
            'balance': balance,
            'burst_mode': BURST_MODE_ACTIVE,
            'hot_streak': wins_in_attack >= 2 if ATTACK_MODE_ACTIVE else False,
            'funding_aligned': funding_aligned
        })

    # =============================
    # RESULTS REPORT
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
    print("OPUS D (DYNAMIC) BACKTEST RESULTS")
    print("="*40)
    print(f"Final Balance:    {balance:.2f}")
    print(f"Max Drawdown:     {max_dd_pct*100:.2f}%")
    print(f"Total Trades:     {len(trades_arr)}")
    
    if len(trades_arr) > 0:
        print(f"Total R:          {trades_arr.sum():.2f}")
        print(f"Avg R:            {trades_arr.mean():.4f}")
        print(f"Win Rate:         {(trades_arr > 0).mean()*100:.1f}%")
        print(f"Sharpe (R-based): {trades_arr.mean() / (trades_arr.std() + 1e-6):.3f}")
    
    print("\nContainment Metrics:")
    print(f"Attack Activations: {stats['attack_activations']}")
    print(f"Burst Activations:  {stats['burst_mode_activations']}")
    print(f"Burst Trades:       {stats['burst_mode_trades']}")
    print(f"Burst Kills:        {stats['burst_mode_kills']}")
    print(f"Kills (Losses):     {stats['attack_kills_loss']}")
    print(f"Kills (Drawdown):   {stats['attack_kills_dd']}")
    print("="*40)
    
    # Save CSV (Disabled for Jul-Nov run)
    # if trade_log:
    #     pd.DataFrame(trade_log).to_csv("chronological_trades.csv", index=False)
    #     print("\nSaved trade log to chronological_trades.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    
    logger = setup_logging()
    run_backtest(args)

if __name__ == "__main__":
    main()
