import pandas as pd
import numpy as np
from datetime import datetime, time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MASTER THINNING STRATEGY - REPRODUCIBLE BASELINE
# ============================================================================
# Template: liv_eq_oct.py (+9.29 bp result)
# ============================================================================

DATA_PATH = 'data/parquet/DOGEUSDT_1m.parquet'

print('='*80)
print('MASTER THINNING STRATEGY - REPRODUCIBLE BASELINE')
print('Data source will be read from DATA_PATH variable')
print('='*80)

# Load data from DATA_PATH
bars = pd.read_parquet(DATA_PATH)

# Data integrity block
print(f'First candle: {bars.index[0]}')
print(f'Last candle: {bars.index[-1]}')
print(f'Total records: {len(bars):,}')

# Derive start_date and end_date from the file
start_date = bars.index.min()
end_date = bars.index.max()
bars = bars.loc[start_date:end_date].copy()
print(f'Using full file range: {bars.index.min()} to {bars.index.max()} ({len(bars):,} bars)')

if len(bars) == 0:
    print('ERROR: No data found in the specified range')
    exit()

# ============================================================================
# 1. CALCULATE MULTI-TIMEFRAME METRICS
# ============================================================================

def calculate_atr(high, low, close, period=20):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Calculate baseline metrics on 1-min bars
bars['range'] = bars['high'] - bars['low']
bars['atr_20'] = calculate_atr(bars['high'], bars['low'], bars['close'], 20)
bars['upper_wick'] = bars['high'] - np.maximum(bars['open'], bars['close'])
bars['lower_wick'] = np.minimum(bars['open'], bars['close']) - bars['low']
bars['volume_median_50'] = bars['volume'].rolling(50).median()
bars['upper_wick_pct'] = bars['upper_wick'] / bars['range']
bars['lower_wick_pct'] = bars['lower_wick'] / bars['range']
bars['upper_wick_pct'] = bars['upper_wick_pct'].replace([np.inf, -np.inf], 0)
bars['lower_wick_pct'] = bars['lower_wick_pct'].replace([np.inf, -np.inf], 0)

# Calculate EMAs for trend
bars['ema_20'] = bars['close'].ewm(span=20).mean()
bars['ema_50'] = bars['close'].ewm(span=50).mean()
bars['trend'] = np.where(bars['ema_20'] > bars['ema_50'], 1, -1)

print('✓ Calculated baseline metrics (ATR, ranges, wicks, EMAs)')

# ============================================================================
# RESAMPLE TO MULTIPLE TIMEFRAMES
# ============================================================================

# 5-minute bars for medium-term context
bars5 = bars.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

bars5['range'] = bars5['high'] - bars5['low']
bars5['atr_20'] = calculate_atr(bars5['high'], bars5['low'], bars5['close'], 20)
bars5['upper_wick'] = bars5['high'] - np.maximum(bars5['open'], bars5['close'])
bars5['lower_wick'] = np.minimum(bars5['open'], bars5['close']) - bars5['low']
bars5['volume_median_50'] = bars5['volume'].rolling(50).median()
bars5['upper_wick_pct'] = bars5['upper_wick'] / bars5['range']
bars5['lower_wick_pct'] = bars5['lower_wick'] / bars5['range']
bars5['upper_wick_pct'] = bars5['upper_wick_pct'].replace([np.inf, -np.inf], 0)
bars5['lower_wick_pct'] = bars5['lower_wick_pct'].replace([np.inf, -np.inf], 0)

# 15-minute bars for higher timeframe structure
bars15 = bars.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

bars15['range'] = bars15['high'] - bars15['low']
bars15['ema_20'] = bars15['close'].ewm(span=20).mean()
bars15['ema_50'] = bars15['close'].ewm(span=50).mean()
bars15['trend'] = np.where(bars15['ema_20'] > bars15['ema_50'], 1, -1)

print(f'✓ Resampled to {len(bars5):,} 5-minute and {len(bars15):,} 15-minute bars')

# ============================================================================
# 2. LIQUIDITY EVENT DETECTION
# ============================================================================

LIQUIDITY_SWEEP_PARAMS = {
    'range_mult': 1.8, 'wick_pct': 0.35, 'volume_mult': 1.5, 'lookback_range': 5
}

LIQUIDATION_CLUSTER_PARAMS = {
    'consecutive_candles': 3, 'cumulative_atr_mult': 2.5, 
    'exhaustion_volume_drop': 0.6, 'failure_to_extend': True
}

LIQUIDITY_THINNING_PARAMS = {
    'range_expansion_mult': 1.6, 'volume_divergence': 0.7,
    'subsequent_wick_threshold': 0.30, 'rapid_alternation': 3
}

def detect_liquidity_sweeps(bars, params):
    sweeps = pd.Series(False, index=bars.index)
    for i in tqdm(range(params['lookback_range'], len(bars)), desc='Liquidity Sweeps', ncols=80):
        current = bars.iloc[i]
        prior_start = max(0, i - params['lookback_range'])
        prior_bars = bars.iloc[prior_start:i]
        
        if len(prior_bars) < params['lookback_range']:
            continue
            
        range_condition = current['range'] >= (params['range_mult'] * current['atr_20'])
        volume_condition = current['volume'] >= (params['volume_mult'] * current['volume_median_50'])
        upper_wick_condition = current['upper_wick_pct'] >= params['wick_pct']
        lower_wick_condition = current['lower_wick_pct'] >= params['wick_pct']
        wick_condition = upper_wick_condition or lower_wick_condition
        
        sweeps.iloc[i] = range_condition and volume_condition and wick_condition
    return sweeps

def detect_liquidation_clusters(bars, params):
    clusters = pd.Series(False, index=bars.index)
    for i in tqdm(range(params['consecutive_candles'], len(bars) - 1), desc='Liquidation Clusters', ncols=80):
        cluster_bars = bars.iloc[i - params['consecutive_candles'] + 1:i + 1]
        bullish_cluster = all(cluster_bars['close'] > cluster_bars['open'])
        bearish_cluster = all(cluster_bars['close'] < cluster_bars['open'])
        
        if not (bullish_cluster or bearish_cluster):
            continue
        
        cluster_start_price = cluster_bars.iloc[0]['open']
        cluster_end_price = cluster_bars.iloc[-1]['close']
        cumulative_move = abs(cluster_end_price - cluster_start_price)
        avg_atr = cluster_bars['atr_20'].mean()
        cumulative_condition = cumulative_move >= (params['cumulative_atr_mult'] * avg_atr)
        
        if not cumulative_condition:
            continue
        
        next_candle = bars.iloc[i + 1]
        cluster_avg_volume = cluster_bars['volume'].mean()
        volume_drop = next_candle['volume'] < (params['exhaustion_volume_drop'] * cluster_avg_volume)
        cluster_avg_range = cluster_bars['range'].mean()
        failure_to_extend = next_candle['range'] < cluster_avg_range
        
        if volume_drop and failure_to_extend:
            clusters.iloc[i + 1] = True
    return clusters

def detect_liquidity_thinning(bars, params):
    thinning_events = pd.Series(False, index=bars.index)
    for i in tqdm(range(10, len(bars) - 3), desc='Liquidity Thinning', ncols=80):
        current = bars.iloc[i]
        prior_bars = bars.iloc[i-10:i]
        avg_prior_range = prior_bars['range'].mean()
        avg_prior_volume = prior_bars['volume'].mean()
        
        range_expansion = current['range'] >= (params['range_expansion_mult'] * avg_prior_range)
        volume_divergence = current['volume'] < (params['volume_divergence'] * avg_prior_volume * params['range_expansion_mult'])
        
        if not (range_expansion and volume_divergence):
            continue
        
        subsequent_bars = bars.iloc[i+1:i+4]
        large_wicks = (subsequent_bars['upper_wick_pct'] >= params['subsequent_wick_threshold']).any() or \
                     (subsequent_bars['lower_wick_pct'] >= params['subsequent_wick_threshold']).any()
        
        thinning_events.iloc[i] = large_wicks
    return thinning_events

# Detect events on 5-minute timeframe
print('\nDetecting liquidity events on 5-minute timeframe...')
bars5['liquidity_sweep'] = detect_liquidity_sweeps(bars5, LIQUIDITY_SWEEP_PARAMS)
bars5['liquidation_cluster'] = detect_liquidation_clusters(bars5, LIQUIDATION_CLUSTER_PARAMS)
bars5['liquidity_thinning'] = detect_liquidity_thinning(bars5, LIQUIDITY_THINNING_PARAMS)
bars5['any_liquidity_event'] = bars5['liquidity_sweep'] | bars5['liquidation_cluster'] | bars5['liquidity_thinning']

event_counts = {
    'Liquidity Sweeps': bars5['liquidity_sweep'].sum(),
    'Liquidation Clusters': bars5['liquidation_cluster'].sum(),
    'Liquidity Thinning': bars5['liquidity_thinning'].sum(),
    'Total Events': bars5['any_liquidity_event'].sum()
}

print('✓ Event detection complete')
for event, count in event_counts.items():
    print(f'  - {event}: {count:,}')

# ============================================================================
# 3. TRADE EXECUTION WITH ENHANCED LOGIC
# ============================================================================

TRADE_PARAMS = {
    'cooldown_minutes': 15,           # Reduced from 30 for more opportunities
    'max_trades_per_day': 8,          # Increased from 5
    'hold_times': [5, 15, 30, 60],   # Multiple hold times
    'transaction_cost': 0.000075,     # 0.75bp per side = 1.5bp round-turn
    'stop_loss_atr_mult': 1.5,        # OPTIMAL: 1.5 ATR stop
    'take_profit_atr_mult': 4.5,      # OPTIMAL: 4.5 ATR take-profit - 3:1 R:R
    'min_atr': 0.15,                  # Minimum volatility filter
    'max_atr': 2.0,                   # Maximum volatility filter
    'trend_filter': True,             # Use trend filter
    'volume_confirmation': True,      # Volume confirmation required
    'session_filter': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # Extended hours
}

print(f"\n{'='*80}")
print('TRADE EXECUTION - ENHANCED STRATEGY')
print(f"{'='*80}")

trade_records = []

# Test multiple hold times and choose the best one
for hold_time in TRADE_PARAMS['hold_times']:
    print(f"\n--- TESTING {hold_time} MINUTE HOLD ---")
    
    temp_records = []
    last_trade_idx = -TRADE_PARAMS['cooldown_minutes']
    current_date = None
    trades_today = 0
    daily_pnl = 0

    for i in tqdm(range(len(bars5) - hold_time), desc=f'Finding {hold_time}-min trades', ncols=80):
        if i - last_trade_idx < TRADE_PARAMS['cooldown_minutes']:
            continue

        bar_date = bars5.index[i].date()
        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0
            daily_pnl = 0

        if trades_today >= TRADE_PARAMS['max_trades_per_day']:
            continue

        # Session filter
        bar_time = bars5.index[i].time()
        if bar_time.hour not in TRADE_PARAMS['session_filter']:
            continue

        # Check for liquidity event in recent bars
        lookback_start = max(0, i - 5)
        recent_events = bars5['any_liquidity_event'].iloc[lookback_start:i+1]
        
        if not recent_events.any():
            continue

        # Get the most recent event
        event_idx = recent_events[recent_events].index[-1]
        event_pos = bars5.index.get_loc(event_idx)
        event_bar = bars5.iloc[event_pos]
        
        # Volatility filter
        current_atr = bars5['atr_20'].iloc[i]
        if current_atr < TRADE_PARAMS['min_atr'] or current_atr > TRADE_PARAMS['max_atr']:
            continue

        # Volume filter
        if TRADE_PARAMS['volume_confirmation']:
            if bars5['volume'].iloc[i] < bars5['volume_median_50'].iloc[i] * 0.8:
                continue

        # Trend filter - trade with higher timeframe trend
        direction = None
        if TRADE_PARAMS['trend_filter']:
            # Get 15-min trend at this time
            current_time = bars5.index[i]
            try:
                trend_15 = bars15.loc[bars15.index <= current_time, 'trend'].iloc[-1]
            except:
                trend_15 = 1  # Default to bullish
        else:
            trend_15 = 1

        # Determine direction based on event type
        if bars5['liquidity_sweep'].iloc[event_pos]:
            # Fade the sweep (trade against the wick)
            if event_bar['upper_wick_pct'] >= LIQUIDITY_SWEEP_PARAMS['wick_pct']:
                direction = -1  # Short after upper wick sweep
            elif event_bar['lower_wick_pct'] >= LIQUIDITY_SWEEP_PARAMS['wick_pct']:
                direction = 1   # Long after lower wick sweep

        elif bars5['liquidation_cluster'].iloc[event_pos]:

            # Fade the cluster (mean reversion)
            cluster_start = max(0, event_pos - LIQUIDATION_CLUSTER_PARAMS['consecutive_candles'])
            cluster_bars = bars5.iloc[cluster_start:event_pos]
            if len(cluster_bars) > 0:
                cluster_start_price = cluster_bars.iloc[0]['open']
                cluster_end_price = cluster_bars.iloc[-1]['close']
                cluster_direction = 1 if cluster_end_price > cluster_start_price else -1
                direction = -cluster_direction  # Fade the cluster

        elif bars5['liquidity_thinning'].iloc[event_pos]:
            # Trade with the breakout after thinning
            subsequent_bars = bars5.iloc[event_pos+1:event_pos+4]
            if len(subsequent_bars) > 0:
                if (subsequent_bars['upper_wick_pct'] >= 0.30).any():
                    direction = 1  # Long if upper wicks
                elif (subsequent_bars['lower_wick_pct'] >= 0.30).any():
                    direction = -1  # Short if lower wicks

        if direction is None:
            continue

        # Apply trend filter
        if TRADE_PARAMS['trend_filter'] and direction != trend_15:
            continue  # Only trade with the higher timeframe trend

        # Determine event type for tracking
        event_type = 'sweep' if bars5['liquidity_sweep'].iloc[event_pos] else \
                    ('cluster' if bars5['liquidation_cluster'].iloc[event_pos] else 'thinning')
        
        # Calculate entry and risk levels - use 1.2 ATR for all (revert failed tweak)
        entry_price = bars5['close'].iloc[i]
        stop_loss = entry_price - (direction * current_atr * TRADE_PARAMS['stop_loss_atr_mult'])
        take_profit = entry_price + (direction * current_atr * TRADE_PARAMS['take_profit_atr_mult'])

        # Check if stop loss is too close
        if direction == 1 and stop_loss >= entry_price * 0.999:
            continue
        if direction == -1 and stop_loss <= entry_price * 1.001:
            continue

        # Simulate trade
        exit_price = None
        exit_reason = None
        
        # Check for stop loss or take profit during hold period
        for j in range(i + 1, min(i + hold_time, len(bars5))):
            current_price = bars5['close'].iloc[j]
            
            if direction == 1:  # LONG
                if current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                elif current_price >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
            else:  # SHORT
                if current_price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                elif current_price <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
        
        # If no exit triggered, exit at end of hold period
        if exit_price is None:
            exit_price = bars5['close'].iloc[i + hold_time]
            exit_reason = 'time'

        # Calculate returns
        raw_return = (exit_price - entry_price) / entry_price * direction
        net_return = raw_return - 2 * TRADE_PARAMS['transaction_cost']
        pnl_bp = net_return * 1e4

        # Daily loss limit
        if daily_pnl + pnl_bp < -100:  # Max -100bp per day
            continue

        temp_records.append({
            'timestamp': bars5.index[i],
            'hold_time': hold_time,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'exit_reason': exit_reason,
            'pnl_bp': pnl_bp,
            'raw_return_bp': raw_return * 1e4,
            'atr': current_atr,
            'event_type': event_type
        })


        last_trade_idx = i
        trades_today += 1
        daily_pnl += pnl_bp

    # Show results for each hold time and keep all trades for aggregate analysis
    if temp_records:
        avg_pnl = np.mean([t['pnl_bp'] for t in temp_records])
        win_rate = len([t for t in temp_records if t['pnl_bp'] > 0]) / len(temp_records)
        print(f"  {hold_time}-min: {len(temp_records)} trades, {avg_pnl:+.2f} bp avg, {win_rate:.1%} win rate")
        
        # Keep all trades to match original methodology
        trade_records.extend(temp_records)


# ============================================================================
# 4. PERFORMANCE ANALYSIS
# ============================================================================

if trade_records:
    trade_df = pd.DataFrame(trade_records)
    
    print(f"\n{'='*80}")
    print(f'FINAL PERFORMANCE SUMMARY - {len(trade_df)} TOTAL TRADES')
    print(f"{'='*80}")

    winning_trades = trade_df[trade_df['pnl_bp'] > 0]
    losing_trades = trade_df[trade_df['pnl_bp'] <= 0]
    
    total_pnl = trade_df['pnl_bp'].sum()
    avg_pnl = trade_df['pnl_bp'].mean()
    win_rate = len(winning_trades) / len(trade_df)
    
    print(f'Win Rate: {win_rate:.1%}')
    print(f'Average P&L: {avg_pnl:+.2f} bp')
    print(f'Total P&L: {total_pnl:+.1f} bp')
    print(f'Best Trade: +{trade_df["pnl_bp"].max():.1f} bp')
    print(f'Worst Trade: {trade_df["pnl_bp"].min():.1f} bp')
    
    # Performance by exit reason
    print(f"\nExit Reason Breakdown:")
    for reason in trade_df['exit_reason'].unique():
        reason_trades = trade_df[trade_df['exit_reason'] == reason]
        print(f"  {reason}: {len(reason_trades)} trades, avg {reason_trades['pnl_bp'].mean():+.2f} bp")
    
    # Performance by event type
    print(f"\nEvent Type Breakdown:")
    for event_type in trade_df['event_type'].unique():
        event_trades = trade_df[trade_df['event_type'] == event_type]
        print(f"  {event_type}: {len(event_trades)} trades, avg {event_trades['pnl_bp'].mean():+.2f} bp")

    print(f"\n{'='*80}")
    print("TOP 20 TRADES (Chronological)")
    print(f"{'='*80}")
    print(f"{'Time':<20} | {'Dir':<5} | {'Event':<8} | {'Hold':<5} | {'Exit':<8} | {'P&L':<7}")
    print(f"{'-'*80}")

    # Show best trades first
    top_trades = trade_df.nlargest(20, 'pnl_bp')
    for _, trade in top_trades.iterrows():
        print(f"{str(trade.timestamp):<20} | {trade.direction:<5} | {trade.event_type:<8} | {trade.hold_time:>4}m | {trade.exit_reason:<8} | {trade.pnl_bp:>+6.1f}")

    # Save results
    output_file = f'data/processed/{DATA_PATH.split("/")[-1].split(".")[0]}_trades_enhanced.csv'
    trade_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed trade log saved: {output_file}")
    
    # Check if targets are met
    print(f"\n{'='*80}")
    print("PERFORMANCE TARGETS:")
    print(f"{'='*80}")
    print(f"Trades: {len(trade_df)} (Target: >60) {'✓' if len(trade_df) > 60 else '✗'}")
    print(f"Avg P&L: {avg_pnl:+.2f} bp (Target: >+3bp) {'✓' if avg_pnl > 3 else '✗'}")
    print(f"Win Rate: {win_rate:.1%} (Target: >45%) {'✓' if win_rate > 0.45 else '✗'}")
    print(f"{'='*80}")

else:
    print("\nNo trades found with current parameters")
    print("Try adjusting parameters to generate more trading opportunities")
