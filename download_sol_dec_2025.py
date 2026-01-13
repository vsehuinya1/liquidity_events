#!/usr/bin/env python3
"""
Quick script to download SOLUSDT December 2025 data
"""

import sys
sys.path.append('/Users/MartinOile/Desktop/liquidity_events')

from download_binance_data import download_month_data, convert_to_dataframe, save_to_parquet

print("Downloading SOLUSDT December 2025 data...")
klines = download_month_data("SOLUSDT", year=2025, month=12)

if klines:
    df = convert_to_dataframe(klines)
    if not df.empty:
        save_to_parquet(df, "SOLUSDT")
        print("✓ Download complete!")
    else:
        print("✗ Failed to convert data")
else:
    print("✗ Failed to download data")
