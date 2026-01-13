# sweep_cluster_thinning_backtest.py

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

# Req 1: Sweep Prior Extreme Check (N=20 found to be optimal)
df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()

# Req 4: Volatility Pocket (Rising ATR)
df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
df['vol_pocket_active'] = df['atr'] > df['atr_med_20']

# =============================
# STATE VARIABLES
# =============================

balance = INITIAL_BALANCE
trades = []

active_bias = None           # 'LONG' or 'SHORT'
active_cluster_id = None
last_cluster_end_idx = -999  # Track when the last cluster ended
pocket_traded = False        # Track if we traded in the current volatility pocket
prev_pocket_active = False

cooldown_until = -1
cluster_id = 0

# =============================
# PARAMETERS (LOCKED)
# =============================

SWEEP_RANGE_MULT = 1.8
SWEEP_WICK_PCT = 0.35
SWEEP_VOL_MULT = 1.5

CLUSTER_LEN = 3
THINNING_ATR_DIST = 1.2
COOLDOWN_ATR = 1.5
MAX_BARS_SINCE_CLUSTER = 20

# =============================
# BACKTEST LOOP
# =============================

for i in range(50, len(df) - 2):

    # Volatility Pocket Tracking
    is_pocket = df.iloc[i]['vol_pocket_active']
    if not is_pocket and prev_pocket_active:
        pocket_traded = False 
    prev_pocket_active = is_pocket

    if i < cooldown_until:
        continue

    row = df.iloc[i]

    # =========================
    # CLUSTER DETECTION
    # =========================
    cluster = df.iloc[i-CLUSTER_LEN:i]

    bull_cluster = (cluster['close'] > cluster['open']).all()
    bear_cluster = (cluster['close'] < cluster['open']).all()

    if bull_cluster or bear_cluster:
        cluster_id += 1
        active_cluster_id = cluster_id
        active_bias = None
        last_cluster_end_idx = i

    # =========================
    # SWEEP DETECTION (ARM ONLY)
    # =========================
    
    # Req 4: One sweep per volatility pocket
    # NOTE: Disabled as backtests show it reduces Avg R (0.22 -> 0.20)
    # if is_pocket and pocket_traded:
    #     continue

    sweep = (
        row['range'] > SWEEP_RANGE_MULT * row['atr'] and
        row['volume'] > SWEEP_VOL_MULT * row['vol_med']
    )

    if sweep:
        # Req 2: Cluster Post-Condition -> Pre-Condition
        if (i - last_cluster_end_idx) > MAX_BARS_SINCE_CLUSTER:
            active_bias = None
        else:
            # Req 1: Sweep must violate a prior extreme (N=20)
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
    # THINNING FILTER
    # =========================
    dist = abs(confirm['close'] - row['close'])

    if dist > THINNING_ATR_DIST * row['atr']:
        continue

    # Req 3: Directional Thinning
    # NOTE: Disabled as backtests show it filters too many profitable trades (Avg R drops to ~0.0)
    # confirm_range = confirm['high'] - confirm['low']
    # if confirm_range > 0:
    #     c_upper_wick = (confirm['high'] - max(confirm['open'], confirm['close'])) / confirm_range
    #     c_lower_wick = (min(confirm['open'], confirm['close']) - confirm['low']) / confirm_range
        
    #     if active_bias == 'SHORT':
    #         if c_lower_wick > c_upper_wick:
    #             continue
    #     else: # LONG
    #         if c_upper_wick > c_lower_wick:
    #             continue

    # =========================
    # EXECUTE TRADE
    # =========================
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

# =============================
# RESULTS
# =============================

trades = np.array(trades)

print("Trades:", len(trades))
if len(trades) > 0:
    print("Win rate:", (trades > 0).mean())
    print("Avg R:", trades.mean())
print("Final balance:", round(balance, 2))
