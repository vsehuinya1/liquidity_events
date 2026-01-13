import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = 'data/parquet/SOLUSDT_1m.parquet'

ATR_PERIOD = 20
CLUSTER_LOOKBACK = 240        # 4h
CLUSTER_TOL_ATR = 0.6
CLUSTER_MIN_TOUCHES = 3

COST_BP = 1.5 / 10000
R_TARGET = 2.0
INITIAL_BALANCE = 10_000
RISK_PER_TRADE = 0.01

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)
df = df.sort_index()

# ============================================================
# INDICATORS
# ============================================================

tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)

df['atr'] = tr.rolling(ATR_PERIOD).mean()
df['vol_med'] = df['volume'].rolling(50).median()

df = df.dropna()

# ============================================================
# CLUSTER DETECTION
# ============================================================

def in_cluster(idx):
    window = df.iloc[idx - CLUSTER_LOOKBACK:idx]
    atr = df.iloc[idx].atr

    highs = window['high']
    lows = window['low']

    level = df.iloc[idx].close
    tol = CLUSTER_TOL_ATR * atr

    touches = (
        ((highs - level).abs() < tol) |
        ((lows - level).abs() < tol)
    ).sum()

    return touches >= CLUSTER_MIN_TOUCHES

# ============================================================
# SWEEP STATE
# ============================================================

sweep_state = {
    "active": False,
    "direction": None,
    "sweep_high": None,
    "sweep_low": None,
    "sweep_range": None,
    "sweep_volume": None,
    "atr": None,
    "bars_since": 0
}

# ============================================================
# SWEEP DETECTOR
# ============================================================

def detect_sweep(bar):
    rng = bar.high - bar.low
    if rng < 1.8 * bar.atr:
        return None

    wick_up = bar.high - max(bar.open, bar.close)
    wick_down = min(bar.open, bar.close) - bar.low

    if max(wick_up, wick_down) / rng < 0.35:
        return None

    if bar.volume < bar.vol_med * 1.5:
        return None

    direction = "up" if wick_up > wick_down else "down"

    return {
        "direction": direction,
        "high": bar.high,
        "low": bar.low,
        "range": rng,
        "volume": bar.volume,
        "atr": bar.atr
    }

# ============================================================
# BACKTEST LOOP
# ============================================================

balance = INITIAL_BALANCE
trades = []

for i in range(max(ATR_PERIOD + 50, CLUSTER_LOOKBACK + 1), len(df)):
    bar = df.iloc[i]

    # ---------------- ACTIVE SWEEP ----------------
    if sweep_state["active"]:
        sweep_state["bars_since"] += 1

        if sweep_state["bars_since"] > 2:
            sweep_state["active"] = False
            continue

        atr = sweep_state["atr"]

        # Continuation invalidation
        if sweep_state["direction"] == "up":
            if bar.high - sweep_state["sweep_high"] > 0.25 * atr:
                sweep_state["active"] = False
                continue
        else:
            if sweep_state["sweep_low"] - bar.low > 0.25 * atr:
                sweep_state["active"] = False
                continue

        # Failure confirmation
        closes_inside = (
            sweep_state["sweep_low"] <= bar.close <= sweep_state["sweep_high"]
        )
        vol_contracts = bar.volume <= sweep_state["sweep_volume"]
        range_contracts = (bar.high - bar.low) < sweep_state["sweep_range"]

        if closes_inside and vol_contracts and range_contracts:
            side = "short" if sweep_state["direction"] == "up" else "long"
            entry = bar.close

            if side == "short":
                stop = sweep_state["sweep_high"] + 0.2 * atr
                risk = stop - entry
            else:
                stop = sweep_state["sweep_low"] - 0.2 * atr
                risk = entry - stop

            size = (balance * RISK_PER_TRADE) / risk
            target = entry + (risk * R_TARGET if side == "long" else -risk * R_TARGET)

            for j in range(i + 1, min(i + 50, len(df))):
                fwd = df.iloc[j]

                hit_stop = (
                    fwd.low <= stop if side == "long"
                    else fwd.high >= stop
                )
                hit_target = (
                    fwd.high >= target if side == "long"
                    else fwd.low <= target
                )

                if hit_stop or hit_target:
                    r_mult = -1 if hit_stop else R_TARGET
                    pnl = balance * RISK_PER_TRADE * r_mult
                    pnl -= abs(size * entry) * COST_BP
                    balance += pnl

                    trades.append({
                        "entry_idx": i,
                        "exit_idx": j,
                        "side": side,
                        "R": r_mult,
                        "balance": balance
                    })
                    break

            sweep_state["active"] = False

        continue

    # ---------------- SWEEP DETECTION ----------------
    if not in_cluster(i):
        continue

    sweep = detect_sweep(bar)
    if sweep:
        sweep_state.update({
            "active": True,
            "direction": sweep["direction"],
            "sweep_high": sweep["high"],
            "sweep_low": sweep["low"],
            "sweep_range": sweep["range"],
            "sweep_volume": sweep["volume"],
            "atr": sweep["atr"],
            "bars_since": 0
        })

# ============================================================
# RESULTS
# ============================================================

trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("No trades.")
else:
    print(f"Trades: {len(trades_df)}")
    print(f"Win rate: {(trades_df.R > 0).mean():.3f}")
    print(f"Avg R: {trades_df.R.mean():.3f}")
    print(f"Final balance: {balance:.2f}")
