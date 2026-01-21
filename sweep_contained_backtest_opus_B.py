# sweep_contained_backtest_opus.py
# OPUS: Optimized Parameters for Ultimate Strategy
# Aggressively tuned for maximum Avg R and Total R

import pandas as pd
import numpy as np
import argparse
import logging
import sys
from datetime import datetime

# =============================
# OPUS CONFIG - AGGRESSIVE OPTIMIZATION
# =============================

INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5  # round-trip
ATR_PERIOD = 20  # Original ATR period for stable signals

# OPUS Containment - OPTION B: ULTRA LOW DRAWDOWN
MAX_DRAWDOWN_SESSION_R = 1.0   # Extremely strict session stop (was 2.0)
MAX_CONSECUTIVE_LOSSES = 2     # Strict loss limit
COOLDOWN_TRADES = 8            # Long cooldown to skip bad regimes (was 5)
REGIME_WINDOW = 12             # More smoothing
MIN_REGIME_EXPECTANCY = 0.1    # Require strict positive edge

# OPUS Sizing - CONSERVATIVE
ATTACK_SIZE_MULT = 1.2         # Slight boost only
HOT_STREAK_SIZE_MULT = 1.4     # Minimal streak bonus
FUNDING_SQUEEZE_BONUS = 0.1    # Minimal funding bonus

# OPUS Signal Parameters - Same as Original
SWEEP_THINNING_MULT = 1.2      
SWEEP_WICK_PCT = 0.35          
MAX_BARS_SINCE_CLUSTER = 20    
CLUSTER_COMPRESSION_RATIO = 0.7  

# OPUS Exit - Tighter to lock profits
ATR_TRAILING_STOP_MULT = 1.6   # Tighter than original (1.8) to reduce deep drawdowns
INITIAL_STOP_ATR = 0.9         # Tighter initial
COOLDOWN_ATR = 1.5             # Original

# Statistics
stats = {
    'attack_activations': 0,
    'attack_kills_dd': 0,
    'attack_kills_loss': 0,
    'attack_regime_blocks': 0,
    'cooldown_activations': 0,
    'total_trades': 0,
    'sized_up_trades': 0,
    'hot_streak_trades': 0,
    'funding_aligned_trades': 0,
    'quality_filtered_out': 0,
    'momentum_filtered_out': 0
}

# =============================
# LOGGING SETUP
# =============================
def setup_logging():
    date_str = datetime.now().strftime("%Y%m%d")
    log_dir = "logs/backtests"
    import os
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/opus_test_{date_str}.log"
    
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
    
    data_path = args.data_path if args and args.data_path else 'data/parquet/SOLUSDT_1m_with_funding.parquet'
    
    # State flags
    ATTACK_MODE_ACTIVE = False
    losses_in_attack = 0
    wins_in_attack = 0  # NEW: Track wins for hot streak
    attack_session_pnl = 0.0
    
    cooldown_counter = 0
    
    # Validation Tracking
    trade_results = []
    consecutive_wins = 0  # NEW: Track win streaks
    
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
    # INDICATORS - OPUS ENHANCED
    # =============================
    df['range'] = df['high'] - df['low']
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    
    # Wick Analysis - Enhanced
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['range'].replace(0, np.nan)
    df['body'] = (df['close'] - df['open']).abs()
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)
    
    # Sweep Lookback - Multi-timeframe
    df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()
    df['roll_max_10'] = df['high'].shift(1).rolling(10, min_periods=1).max()
    df['roll_min_10'] = df['low'].shift(1).rolling(10, min_periods=1).min()
    
    # Volatility Pocket - Enhanced
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['atr_percentile'] = df['atr'].rolling(50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    # NEW: Strong volatility pocket (top 30% of recent ATR)
    df['strong_vol_pocket'] = df['vol_pocket_active'] & (df['atr_percentile'] > 0.7)
    
    # Cluster Helpers
    df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
    df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()
    
    # NEW: Momentum Indicators
    df['close_change_5'] = df['close'].pct_change(5)
    
    # NEW: Volume Profile (if volume exists)
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['vol_spike'] = df['volume'] > 1.5 * df['vol_ma']
    else:
        df['vol_spike'] = True  # Assume always valid if no volume data
    
    # NEW: Funding Rate Momentum
    df['funding_ma'] = df['fundingRate'].rolling(8, min_periods=1).mean()
    df['funding_extreme'] = df['fundingRate'].abs() > df['fundingRate'].abs().rolling(100, min_periods=20).quantile(0.8)

    # =============================
    # SIMULATION LOOP - OPUS LOGIC
    # =============================
    balance = INITIAL_BALANCE
    active_bias = None
    last_cluster_end_idx = -999
    
    cooldown_until_idx = -1
    last_trade_pnl = 0
    recent_loss = False

    print("Starting OPUS simulation with enhanced containment...")
    
    for i in range(50, len(df) - 2):
        row = df.iloc[i]
        timestamp = row.name
        is_pocket = row['vol_pocket_active']
        is_strong_pocket = row['strong_vol_pocket'] if not pd.isna(row['strong_vol_pocket']) else False
        
        # -------------------------
        # ATTACK MODE STATE MANAGEMENT - OPUS ENHANCED
        # -------------------------
        
        # 1. Damage-Based Invalidation
        kill_reason = None
        if ATTACK_MODE_ACTIVE:
            if losses_in_attack >= MAX_CONSECUTIVE_LOSSES:
                kill_reason = f"{MAX_CONSECUTIVE_LOSSES} consecutive losses"
                stats['attack_kills_loss'] += 1
                
            elif attack_session_pnl <= -MAX_DRAWDOWN_SESSION_R:
                kill_reason = f"Session DD {attack_session_pnl:.2f}R hit cap"
                stats['attack_kills_dd'] += 1
                
            if kill_reason:
                ATTACK_MODE_ACTIVE = False
                cooldown_counter = COOLDOWN_TRADES
                stats['cooldown_activations'] += 1
                consecutive_wins = 0  # Reset streak
                wins_in_attack = 0

        # 2. Activation Logic - OPUS: Require stronger conditions
        if not ATTACK_MODE_ACTIVE:
            if cooldown_counter > 0:
                pass
            else:
                # OPUS: Activate on vol pocket with no recent loss (simplified)
                activation_condition = (
                    is_pocket 
                    and last_trade_pnl >= 0 
                    and not recent_loss
                )
                
                if activation_condition:
                    # Regime Governor Check - OPUS: Require positive edge
                    regime_ok = True
                    if len(trade_results) >= REGIME_WINDOW:
                        rolling_R = np.mean(trade_results[-REGIME_WINDOW:])
                        if rolling_R < MIN_REGIME_EXPECTANCY:
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
        # CLUSTER DETECTION - OPUS: Tighter Compression
        # -------------------------
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
            is_cluster = False
        else:
            is_cluster = tr_cluster <= CLUSTER_COMPRESSION_RATIO * tr_prior

        if is_cluster:
            active_bias = None
            last_cluster_end_idx = i

        # -------------------------
        # SWEEP DETECTION - OPUS: Quality Filters
        # -------------------------
        is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        bars_since_cluster = i - last_cluster_end_idx
        
        if is_thin_enough and bars_since_cluster <= MAX_BARS_SINCE_CLUSTER:
            prior_max = row['roll_max_20']
            prior_min = row['roll_min_20']
            
            # OPUS: Also check shorter-term levels for confluence (Unused currently)
            # prior_max_10 = row['roll_max_10']
            # prior_min_10 = row['roll_min_10']
            
            is_valid_short = (
                row['upper_wick_pct'] > SWEEP_WICK_PCT 
                and row['high'] > prior_max 
                and row['close'] < prior_max
            )
            is_valid_long = (
                row['lower_wick_pct'] > SWEEP_WICK_PCT 
                and row['low'] < prior_min 
                and row['close'] > prior_min
            )
            
            if is_pocket:  # Require vol pocket for entry
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
        # OPUS SIZING - MULTI-TIER
        # -------------------------
        
        position_size_mult = 1.0
        
        # Tier 1: Attack Mode Active
        if ATTACK_MODE_ACTIVE:
            position_size_mult = ATTACK_SIZE_MULT
            stats['sized_up_trades'] += 1
            
            # Tier 2: Hot Streak Bonus (2+ wins in attack session)
            if wins_in_attack >= 2:
                position_size_mult = HOT_STREAK_SIZE_MULT
                stats['hot_streak_trades'] += 1
        
        # Tier 3: Funding Alignment Bonus
        funding = row['fundingRate']
        funding_aligned = False
        if active_bias == 'LONG' and funding < -0.0001:  # Shorts paying longs
            funding_aligned = True
        elif active_bias == 'SHORT' and funding > 0.0001:  # Longs paying shorts
            funding_aligned = True
            
        if funding_aligned:
            position_size_mult += FUNDING_SQUEEZE_BONUS
            stats['funding_aligned_trades'] += 1
            
        # -------------------------
        # EXECUTE TRADE - OPUS TIGHTER STOPS
        # -------------------------
        entry = confirm['close']
        atr = row['atr']
        
        if active_bias == 'LONG': 
            stop = entry - INITIAL_STOP_ATR * atr
        else: 
            stop = entry + INITIAL_STOP_ATR * atr

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
                # Trail
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
        
        # Update consecutive wins
        if R_final > 0:
            consecutive_wins += 1
        else:
            consecutive_wins = 0
        
        # Update Containment Counters
        if ATTACK_MODE_ACTIVE:
            attack_session_pnl += R_final
            if R_final < 0:
                losses_in_attack += 1
                wins_in_attack = 0  # Reset win streak in attack
            else:
                losses_in_attack = 0
                wins_in_attack += 1
                
        # Update Cooldown Counter
        if cooldown_counter > 0:
            cooldown_counter -= 1

    # =============================
    # RESULTS REPORT - OPUS ENHANCED
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
    print("OPUS BACKTEST RESULTS")
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
    print(f"Kills (Losses):     {stats['attack_kills_loss']}")
    print(f"Kills (Drawdown):   {stats['attack_kills_dd']}")
    print(f"Regime Blocks:      {stats['attack_regime_blocks']}")
    print(f"Cooldowns:          {stats['cooldown_activations']}")
    print(f"Sized Up Trades:    {stats['sized_up_trades']}")
    
    print("\nOPUS Enhancements:")
    print(f"Hot Streak Trades:  {stats['hot_streak_trades']}")
    print(f"Funding Aligned:    {stats['funding_aligned_trades']}")
    print(f"Quality Filtered:   {stats['quality_filtered_out']}")
    print(f"Momentum Filtered:  {stats['momentum_filtered_out']}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description='Run OPUS Optimized Backtest')
    parser.add_argument('--data_path', type=str, help='Path to the parquet data file')
    args = parser.parse_args()
    
    logger = setup_logging()
    run_backtest(args)

if __name__ == "__main__":
    main()
