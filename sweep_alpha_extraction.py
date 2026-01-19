# sweep_alpha_extraction.py
"""
ALPHA EXTRACTION ENGINE (Phase 1)
---------------------------------
Objective: Decompose edge by Session, Setup Subtype, and Regime.
Output: `analysis/alpha_attribution.csv` containing rich metadata for every trade.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime

# CONFIG
DATA_PATH = 'data/parquet/SOLUSDT_1m_with_funding.parquet'
OUTPUT_FILE = 'analysis/alpha_attribution.csv'
ATR_PERIOD = 20

def get_session(timestamp):
    h = timestamp.hour
    if 0 <= h < 8: return "Asia"
    if 8 <= h < 13: return "London"
    if 13 <= h < 21: return "NY"
    return "Dead"

def setup_logging():
    os.makedirs("analysis", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    return logging.getLogger()

def run_extraction():
    logger = setup_logging()
    
    # 1. LOAD DATA
    logger.info(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH).copy()
    except FileNotFoundError:
        logger.error("Data file not found.")
        return

    df.sort_index(inplace=True)
    if 'fundingRate' not in df.columns: df['fundingRate'] = 0.0

    # 2. INDICATORS (Standard)
    df['range'] = df['high'] - df['low']
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    
    # Wicks
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['range']
    df['lower_wick_pct'] = df['lower_wick'] / df['range']
    
    # Clusters
    df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
    df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()
    
    # Extremes
    df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
    df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()
    
    # Pocket Matches
    df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
    df['atr_slope'] = df['atr'].diff(5)
    df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)
    
    # 3. REGIME METRICS (For Extraction)
    # ATR Terciles (Global or Rolling?) User implies regime Context.
    # We'll use Rolling 5-day percentile to be adaptive, or Global?
    # User said "ATR vs 60-bar distribution".
    df['atr_rank_60'] = df['atr'].rolling(60).rank(pct=True)
    
    # Trend State (20-bar slope)
    # Simple Linreg slope or just diff? User said "20-bar slope".
    # We use (Close - Close[20]) / ATR as normalized slope
    df['trend_strength_20'] = (df['close'] - df['close'].shift(20)) / df['atr']
    
    # 4. SIMULATION LOOP (Attribution Only - Logic Matches Backtest)
    SWEEP_THINNING_MULT = 1.2
    SWEEP_WICK_PCT = 0.35
    MAX_BARS_SINCE_CLUSTER = 20
    ATR_TRAILING_STOP_MULT = 1.8
    INITIAL_STOP_ATR = 1.0
    
    trades = []
    last_cluster_end_idx = -999
    
    logger.info("Extracting Trade Metadata...")
    
    for i in range(60, len(df) - 100):
        row = df.iloc[i]
        timestamp = row.name
        
        # Cluster Detection
        tr_cluster = row['tr_roll_cluster']
        tr_prior = row['tr_roll_prior']
        is_cluster = False
        if not (pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0):
            is_cluster = tr_cluster <= 0.7 * tr_prior
            
        if is_cluster:
            last_cluster_end_idx = i
            
        # Sweep Detection
        # 1. Thinning
        is_thin = row['range'] <= SWEEP_THINNING_MULT * row['atr']
        bars_since_cluster = i - last_cluster_end_idx
        
        if not (is_thin and bars_since_cluster <= MAX_BARS_SINCE_CLUSTER):
            continue
            
        # 2. Pocket (Context)
        is_pocket = row['vol_pocket_active']
        if not is_pocket: continue
        
        # 3. Setup Logic
        prior_max = row['roll_max_20']
        prior_min = row['roll_min_20']
        
        active_bias = None
        wick_type = None
        
        is_valid_short = (row['upper_wick_pct'] > SWEEP_WICK_PCT and row['high'] > prior_max and row['close'] < prior_max)
        is_valid_long = (row['lower_wick_pct'] > SWEEP_WICK_PCT and row['low'] < prior_min and row['close'] > prior_min)
        
        if is_valid_short: 
            active_bias = 'SHORT'
            wick_type = 'upper'
        elif is_valid_long: 
            active_bias = 'LONG'
            wick_type = 'lower'
            
        if not active_bias: continue
        
        # 4. Confirmation
        confirm = df.iloc[i + 1]
        failed = False
        if active_bias == 'SHORT': failed = confirm['close'] < row['close']
        else: failed = confirm['close'] > row['close']
        
        if not failed: continue
        
        # 5. Trade Outcome Calculation (Standard Result)
        entry = confirm['close']
        atr = row['atr']
        if active_bias == 'LONG': stop = entry - INITIAL_STOP_ATR * atr
        else: stop = entry + INITIAL_STOP_ATR * atr
        
        exit_price = None
        R_realized = 0
        mdd_r_trade = 0 # Max drawdown during trade in R
        
        for j in range(i + 2, len(df)):
            curr = df.iloc[j]
            hi, lo, cl = curr['high'], curr['low'], curr['close']
            
            # MAE Calculation
            if active_bias == 'LONG':
                mae_price = df.iloc[i+2:j+1]['low'].min()
                current_dd_r = (entry - mae_price) / atr
            else:
                mae_price = df.iloc[i+2:j+1]['high'].max()
                current_dd_r = (mae_price - entry) / atr
            
            if current_dd_r > mdd_r_trade: mdd_r_trade = current_dd_r

            # Trailing Stop outcome
            if active_bias == 'LONG':
                if lo <= stop:
                    exit_price = stop
                    R_realized = (exit_price - entry) / atr
                    break
                new_stop = cl - 1.8 * curr['atr']
                stop = max(stop, new_stop)
            else:
                if hi >= stop:
                    exit_price = stop
                    R_realized = (entry - exit_price) / atr
                    break
                new_stop = cl + 1.8 * curr['atr']
                stop = min(stop, new_stop)
                
            if j - i > 500: break # Safety break
            
        if exit_price is None: continue
        
        # 6. ATTRIBUTION DATA STORE
        session = get_session(confirm.name)
        
        # Ranges
        range_to_atr = row['range'] / row['atr']
        
        # Regimes
        atr_rank = row['atr_rank_60']
        if atr_rank < 0.33: atr_regime = 'LOW'
        elif atr_rank < 0.66: atr_regime = 'MID'
        else: atr_regime = 'HIGH'
        
        trend_val = row['trend_strength_20']
        if abs(trend_val) < 1.0: trend_state = 'FLAT'
        elif trend_val > 0: trend_state = 'UP'
        else: trend_state = 'DOWN'
        
        funding = row['fundingRate']
        if funding > 0.0001: funding_sign = 'POS_HIGH'
        elif funding < -0.0001: funding_sign = 'NEG_HIGH'
        else: funding_sign = 'NEUTRAL'
        
        pocket_strength = row['atr_slope'] # Raw slope magnitude
        
        trade_record = {
            'timestamp': confirm.name,
            'direction': active_bias,
            'R_realized': round(R_realized, 2),
            'MDD_R': round(mdd_r_trade, 2),
            # Dimensions
            'session': session,
            'wick_type': wick_type,
            'range_to_atr': round(range_to_atr, 2),
            'bars_since_cluster': bars_since_cluster,
            'atr_percentile': round(atr_rank, 2),
            'atr_regime': atr_regime,
            'trend_state': trend_state,
            'funding_sign': funding_sign,
            'pocket_strength': round(pocket_strength, 4)
        }
        trades.append(trade_record)
        
    # SAVE
    res_df = pd.DataFrame(trades)
    res_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"DONE. Extracted {len(res_df)} trades to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_extraction()
