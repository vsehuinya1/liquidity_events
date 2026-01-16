import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================

DATA_PATH = 'data/parquet/SOLUSDT_1m.parquet'
INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01
COST_BP = 1.5
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
df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
df['atr_slope'] = df['atr'].diff(5)

df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_wick_pct'] = df['upper_wick'] / df['range']
df['lower_wick_pct'] = df['lower_wick'] / df['range']

df['vol_med'] = df['volume'].rolling(50, min_periods=1).median()

df['roll_max_20'] = df['high'].shift(1).rolling(20, min_periods=1).max()
df['roll_min_20'] = df['low'].shift(1).rolling(20, min_periods=1).min()

# =============================
# PARAMETERS (STRUCTURAL)
# =============================

CLUSTER_LEN = 6
CLUSTER_RANGE_SHRINK = 0.7

SWEEP_RANGE_MULT = 1.8
SWEEP_VOL_MULT = 1.5
SWEEP_WICK_PCT = 0.35

THINNING_ATR_MAX = 1.2
TRAIL_ATR = 1.2

MAX_BARS_SINCE_CLUSTER = 20

# =============================
# STATE
# =============================

balance = INITIAL_BALANCE
trades = []

active_bias = None
last_cluster_idx = -999
cooldown_until = -1

# =============================
# BACKTEST LOOP
# =============================

for i in range(60, len(df) - 2):

    if i < cooldown_until:
        continue

    row = df.iloc[i]

    # =========================
    # CLUSTER = RANGE COMPRESSION
    # =========================

    cluster = df.iloc[i - CLUSTER_LEN:i]
    cluster_range = (cluster['high'] - cluster['low']).mean()
    prev_range = df.iloc[i - CLUSTER_LEN*2:i - CLUSTER_LEN]['range'].mean()

    if cluster_range < CLUSTER_RANGE_SHRINK * prev_range:
        last_cluster_idx = i
        active_bias = None

    # =========================
    # SWEEP ELIGIBILITY
    # =========================

    if (i - last_cluster_idx) > MAX_BARS_SINCE_CLUSTER:
        continue

    if row['range'] < SWEEP_RANGE_MULT * row['atr']:
        continue

    if row['volume'] < SWEEP_VOL_MULT * row['vol_med']:
        continue

    # ATR pocket gating: early expansion only
    if not (row['atr'] > row['atr_med_20'] and row['atr_slope'] > 0):
        continue

    # =========================
    # PRIOR EXTREME VIOLATION + THINNING
    # =========================

    active_bias = None

    if (
        row['high'] > row['roll_max_20'] and
        row['close'] < row['roll_max_20'] and
        row['upper_wick_pct'] > SWEEP_WICK_PCT and
        row['range'] < THINNING_ATR_MAX * row['atr']
    ):
        active_bias = 'SHORT'

    if (
        row['low'] < row['roll_min_20'] and
        row['close'] > row['roll_min_20'] and
        row['lower_wick_pct'] > SWEEP_WICK_PCT and
        row['range'] < THINNING_ATR_MAX * row['atr']
    ):
        active_bias = 'LONG'

    if active_bias is None:
        continue

    # =========================
    # FAILURE CONFIRMATION
    # =========================

    confirm = df.iloc[i + 1]

    if active_bias == 'SHORT' and confirm['close'] >= row['close']:
        continue

    if active_bias == 'LONG' and confirm['close'] <= row['close']:
        continue

    # =========================
    # EXECUTION
    # =========================

    entry = confirm['close']
    atr = row['atr']

    if active_bias == 'LONG':
        stop = entry - atr
    else:
        stop = entry + atr

    trailing_stop = stop
    exit_price = None
    R = 0

    for j in range(i + 2, len(df)):
        bar = df.iloc[j]

        if active_bias == 'LONG':
            trailing_stop = max(trailing_stop, bar['close'] - TRAIL_ATR * atr)
            if bar['low'] <= trailing_stop:
                exit_price = trailing_stop
                R = (exit_price - entry) / atr
                break
        else:
            trailing_stop = min(trailing_stop, bar['close'] + TRAIL_ATR * atr)
            if bar['high'] >= trailing_stop:
                exit_price = trailing_stop
                R = (entry - exit_price) / atr
                break

    if exit_price is None:
        continue

    R -= COST_BP / 10_000
    balance += balance * RISK_PER_TRADE * R
    trades.append(R)

    cooldown_until = j + int(atr)

# =============================
# RESULTS
# =============================

trades = np.array(trades)

print("Trades:", len(trades))
if len(trades):
    print("Win rate:", (trades > 0).mean())
    print("Avg R:", trades.mean())
print("Final balance:", round(balance, 2))
