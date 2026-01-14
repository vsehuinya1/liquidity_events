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

# Req 4: Volatility Pocket (Expanding ATR)
# "ATR > median AND ATR slope > 0"
df['atr_med_20'] = df['atr'].rolling(20, min_periods=1).median()
df['atr_slope'] = df['atr'].diff(5) # Simple slope proxy over 5 bars
df['vol_pocket_active'] = (df['atr'] > df['atr_med_20']) & (df['atr_slope'] > 0)

# Req 1: Cluster Compression Helpers
# "Mean true range of those bars is ≤ 70% of the mean true range of the prior N bars"
# N = CLUSTER_LEN (defined below as 5)
df['tr_roll_cluster'] = tr.rolling(5, min_periods=5).mean()
df['tr_roll_prior'] = tr.shift(5).rolling(5, min_periods=5).mean()

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

SWEEP_THINNING_MULT = 1.2
SWEEP_WICK_PCT = 0.35
# SWEEP_VOL_MULT = 1.5  # Removed per new criteria (implicit)

CLUSTER_LEN = 5
# THINNING_ATR_DIST = 1.2 # Moved to SWEEP_THINNING_MULT
COOLDOWN_ATR = 1.5
MAX_BARS_SINCE_CLUSTER = 20

ATR_TRAILING_STOP_MULT = 1.8
INITIAL_STOP_ATR = 1.0

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
    # CLUSTER DETECTION (COMPRESSION)
    # =========================
    # Req 1: Clusters are liquidity loading zones, not momentum.
    # Mean true range of those bars is ≤ 70% of the mean true range of the prior N bars
    
    tr_cluster = row['tr_roll_cluster']
    tr_prior = row['tr_roll_prior']
    
    # Ensure we have data
    if pd.isna(tr_cluster) or pd.isna(tr_prior) or tr_prior == 0:
        is_cluster = False
    else:
        is_cluster = tr_cluster <= 0.7 * tr_prior

    if is_cluster:
        cluster_id += 1
        active_cluster_id = cluster_id
        active_bias = None # Cluster only arms eligibility
        last_cluster_end_idx = i

    # =========================
    # SWEEP DETECTION (ARM ONLY)
    # =========================
    
    # Req 2: Thinning is not an entry filter. Apply thinning only when deciding whether a sweep is valid.
    # Sweep candle total range must be ≤ X * ATR (default 1.2)
    
    is_thin_enough = row['range'] <= SWEEP_THINNING_MULT * row['atr']
    
    if is_thin_enough:
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

            # Volatility Pocket Check (Req 3: Allow sweeps only during early volatility expansion)
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
    # THINNING FILTER
    # =========================
    # Req 2: Do not apply thinning after confirmation or during entry. 
    # Logic moved to Sweep Eligibility.
    pass

    # =========================
    # EXECUTE TRADE
    # =========================
    #Req 4: ATR Trailing Exit
    
    entry = confirm['close']
    atr = row['atr'] # Entry ATR
    
    if active_bias == 'LONG':
        stop = entry - INITIAL_STOP_ATR * atr
    else:
        stop = entry + INITIAL_STOP_ATR * atr

    exit_price = None
    R = 0

    for j in range(i + 2, len(df)):
        current_bar = df.iloc[j]
        hi = current_bar['high']
        lo = current_bar['low']
        cl = current_bar['close']
        
        # Trailing Stop Calculation (updates on CLOSE of previous bar, applied to CURRENT bar)
        # Note: stop logic logic was initialized before loop.
        # We check for breach, THEN update stop for NEXT bar.
        
        if active_bias == 'LONG':
            # Check Stop Breach
            if lo <= stop:
                exit_price = stop
                R = (exit_price - entry) / (entry - (entry - atr)) # (Exit - Entry) / Risk. Risk = 1 ATR approx
                # Normalized R calc: Risk was 1 ATR at entry? 
                # User says: "Initial stop = 1 ATR from entry"
                # R = (Exit - Entry) / Initial_Risk_Amt
                initial_risk = atr
                R = (exit_price - entry) / initial_risk
                break
            
            # Update Trail
            new_stop = cl - ATR_TRAILING_STOP_MULT * current_bar['atr']
            stop = max(stop, new_stop)
            
        else: # SHORT
            # Check Stop Breach
            if hi >= stop:
                exit_price = stop
                initial_risk = atr
                R = (entry - exit_price) / initial_risk
                break
                
            # Update Trail
            new_stop = cl + ATR_TRAILING_STOP_MULT * current_bar['atr']
            stop = min(stop, new_stop)

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
