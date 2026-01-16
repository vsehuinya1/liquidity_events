import pandas as pd
import numpy as np

# Load files
ohlcv_path = 'data/parquet/SOLUSDT_1m.parquet' # Assuming this is Dec 2025 data as per user context
funding_path = 'data/parquet/SOLUSDT_funding_dec2025.parquet'
output_path = 'data/parquet/SOLUSDT_1m_with_funding.parquet'

print(f"Loading OHLCV from {ohlcv_path}...")
df_ohlcv = pd.read_parquet(ohlcv_path)

print(f"Loading Funding from {funding_path}...")
df_funding = pd.read_parquet(funding_path)

# Verify timezones
if df_ohlcv.index.tz is None:
    df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
if df_funding.index.tz is None:
    df_funding.index = df_funding.index.tz_localize('UTC')

# Sort
df_ohlcv.sort_index(inplace=True)
df_funding.sort_index(inplace=True)

print(f"OHLCV Range: {df_ohlcv.index.min()} to {df_ohlcv.index.max()}")
print(f"Funding Range: {df_funding.index.min()} to {df_funding.index.max()}")

# Merge
# Funding rates are sparse (every 8 hours). We want to forward fill them
# so that every candle knows the *current* active funding rate (or most recent).
# We join, then fill.

print("Merging...")
# Reindex funding to OHLCV index, forward filling
# method='ffill' propagates last valid observation forward
df_merged = df_ohlcv.join(df_funding, how='left')
df_merged['fundingRate'] = df_merged['fundingRate'].ffill()

# Fill initial NaN if any (use 0 or first valid)
df_merged['fundingRate'] = df_merged['fundingRate'].fillna(0)

print("Sample check:")
print(df_merged[['close', 'fundingRate']].head())
print(df_merged[['close', 'fundingRate']].tail())

print(f"Saving to {output_path}...")
df_merged.to_parquet(output_path)
print("Done.")
