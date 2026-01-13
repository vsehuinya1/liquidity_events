# sweep_optimizer.py

import pandas as pd
import numpy as np

DATA_PATH = 'data/parquet/SOLUSDT_1m.parquet'
INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5  # round-trip
ATR_PERIOD = 20

# =============================
# LOAD DATA
# =============================

df = pd.read_parquet(DATA_PATH).copy()
df.sort_index(inplace=True)

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

df['vol_med'] = df['volume'].rolling(50, min_periods=1).median()
df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
df['vol_pocket_active'] = df['atr'] > df['atr_med_20']

# Locked Params
SWEEP_RANGE_MULT = 1.8
SWEEP_WICK_PCT = 0.35
SWEEP_VOL_MULT = 1.5

CLUSTER_LEN = 3
THINNING_ATR_DIST = 1.2
COOLDOWN_ATR = 1.5

def run_backtest(n_lookback, cluster_timeout):
    # Pre-calc rolling extremes for this N
    df['roll_max'] = df['high'].shift(1).rolling(n_lookback, min_periods=1).max()
    df['roll_min'] = df['low'].shift(1).rolling(n_lookback, min_periods=1).min()

    balance = INITIAL_BALANCE
    trades = []
    
    active_bias = None
    active_cluster_id = None
    last_cluster_end_idx = -999
    pocket_traded = False
    prev_pocket_active = False

    cooldown_until = -1
    cluster_id = 0

    for i in range(50, len(df) - 2):
        is_pocket = df.iloc[i]['vol_pocket_active']
        if not is_pocket and prev_pocket_active:
            pocket_traded = False
        prev_pocket_active = is_pocket

        if i < cooldown_until:
            continue

        row = df.iloc[i]

        # Cluster
        cluster = df.iloc[i-CLUSTER_LEN:i]
        bull_cluster = (cluster['close'] > cluster['open']).all()
        bear_cluster = (cluster['close'] < cluster['open']).all()

        if bull_cluster or bear_cluster:
            cluster_id += 1
            active_cluster_id = cluster_id
            active_bias = None
            last_cluster_end_idx = i

        # Sweep (One per pocket)
        if is_pocket and pocket_traded:
            continue

        sweep = (
            row['range'] > SWEEP_RANGE_MULT * row['atr'] and
            row['volume'] > SWEEP_VOL_MULT * row['vol_med']
        )

        if sweep:
            if (i - last_cluster_end_idx) > cluster_timeout:
                active_bias = None
            else:
                prior_max = row['roll_max']
                prior_min = row['roll_min']
                
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

                if is_valid_short:
                    active_bias = 'SHORT'
                elif is_valid_long:
                    active_bias = 'LONG'

        if active_bias is None:
            continue

        confirm = df.iloc[i + 1]

        if active_bias == 'SHORT':
            failed = confirm['close'] < row['close']
        else:
            failed = confirm['close'] > row['close']

        if not failed:
            continue

        dist = abs(confirm['close'] - row['close'])
        if dist > THINNING_ATR_DIST * row['atr']:
            continue

        confirm_range = confirm['high'] - confirm['low']
        if confirm_range > 0:
            c_upper_wick = (confirm['high'] - max(confirm['open'], confirm['close'])) / confirm_range
            c_lower_wick = (min(confirm['open'], confirm['close']) - confirm['low']) / confirm_range
            
            if active_bias == 'SHORT':
                if c_lower_wick > c_upper_wick:
                    continue
            else: 
                if c_upper_wick > c_lower_wick:
                    continue

        entry = confirm['close']
        atr = row['atr']

        if active_bias == 'LONG':
            stop = entry - atr
            target = entry + 3 * atr
        else:
            stop = entry + atr
            target = entry - 3 * atr

        exit_price = None
        R = 0

        for j in range(i + 2, len(df)):
            hi = df.iloc[j]['high']
            lo = df.iloc[j]['low']

            if active_bias == 'LONG':
                if lo <= stop:
                    exit_price = stop
                    R = -1
                    break
                if hi >= target:
                    exit_price = target
                    R = 3
                    break
            else:
                if hi >= stop:
                    exit_price = stop
                    R = -1
                    break
                if lo <= target:
                    exit_price = target
                    R = 3
                    break

        if exit_price is None:
            continue

        cost = COST_BP / 10_000
        R -= cost
        balance += balance * RISK_PER_TRADE * R
        trades.append(R)
        
        if is_pocket:
            pocket_traded = True

        cooldown_until = j + int(COOLDOWN_ATR * atr)
        active_bias = None

    trades_arr = np.array(trades)
    if len(trades_arr) == 0:
        return 0, 0, 0
    return len(trades_arr), (trades_arr > 0).mean(), trades_arr.mean()

print("N_Lookback, Cluster_Timeout, Trades, Win_Rate, Avg_R")
for n in [5, 10, 20, 30, 50]:
    for t in [5, 10, 20, 30, 50, 100]:
        t_cnt, win, avg_r = run_backtest(n, t)
        print(f"{n}, {t}, {t_cnt}, {win:.4f}, {avg_r:.4f}")
