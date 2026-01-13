#!/usr/bin/env python3
"""
Download Binance Futures (Perpetual) OHLCV data for DOGEUSDT and ETHUSDT
for October 2024 and convert to parquet format.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import sys

# Binance Futures API endpoint
BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/klines"

# Rate limiting parameters
RATE_LIMIT_DELAY = 0.06  # ~16 requests per second
MAX_RETRIES = 3

# October 2024 date range
START_DATE = datetime(2024, 10, 1, 0, 0, 0)
END_DATE = datetime(2024, 10, 31, 23, 59, 59)

# Symbols to download
SYMBOLS = ["SOLUSDT"]

# Output directory
OUTPUT_DIR = "data/parquet"


def get_klines(symbol, interval, start_time, end_time, limit=1500):
    """
    Fetch klines (candlestick data) from Binance Futures API
    
    Parameters:
    - symbol: Trading pair (e.g., "DOGEUSDT")
    - interval: Time interval (e.g., "1m")
    - start_time: Start timestamp in milliseconds
    - end_time: End timestamp in milliseconds
    - limit: Maximum number of records per request (max 1500)
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL + ENDPOINT, params=params, timeout=30)
            
            if response.status_code == 429:
                print(f"  Rate limited (429). Waiting {RATE_LIMIT_DELAY * 2}s...")
                time.sleep(RATE_LIMIT_DELAY * 2)
                continue
            
            if response.status_code != 200:
                print(f"  Error {response.status_code}: {response.text}")
                time.sleep(RATE_LIMIT_DELAY)
                continue
            
            data = response.json()
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
    
    return []


def download_month_data(symbol, year=2024, month=10):
    """
    Download all 1-minute data for a specific month
    """
    print(f"\n{'='*70}")
    print(f"Downloading {symbol} 1m data for {year}-{month:02d}")
    print(f"{'='*70}")
    
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
    
    print(f"Date range: {start_date} to {end_date}")
    
    all_klines = []
    current_start = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    # Calculate total expected records for progress bar
    # 1 minute intervals = 1440 records per day
    total_days = (end_date - start_date).days + 1
    total_expected = total_days * 1440
    
    pbar = tqdm(total=total_expected, desc=f"Downloading {symbol}", unit="records")
    
    while current_start < end_timestamp:
        batch_end = min(current_start + (1500 * 60 * 1000), end_timestamp)
        
        klines = get_klines(symbol, "1m", current_start, batch_end, limit=1500)
        
        if not klines:
            print(f"  No data returned for range")
            break
        
        all_klines.extend(klines)
        pbar.update(len(klines))
        
        # Check if we've reached the end
        if len(klines) < 1500:
            break
        
        # Move to next batch
        current_start = klines[-1][0] + 60000  # Add 1 minute in ms
    
    pbar.close()
    
    print(f"\n✓ Downloaded {len(all_klines):,} records for {symbol}")
    
    return all_klines


def convert_to_dataframe(klines):
    """
    Convert Binance klines to pandas DataFrame with proper schema
    """
    if not klines:
        return pd.DataFrame()
    
    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time, 
    #  quote_asset_volume, number_of_trades, taker_buy_base, taker_buy_quote, ignore]
    
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to appropriate types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Convert price and volume columns to float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = df[col].astype(float)
    
    # Set index to open_time and make it timezone-aware (UTC) to match liq_ev.py schema
    df.set_index('open_time', inplace=True)
    df.index.name = 'timestamp'
    df.index = df.index.tz_localize('UTC')
    
    # Keep only OHLCV columns to match expected schema
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df


def save_to_parquet(df, symbol):
    """
    Save DataFrame to parquet format
    """
    if df.empty:
        print(f"✗ No data to save for {symbol}")
        return False
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create filename (convert DOGEUSDT to DOGEUSDT_1m)
    filename = f"{OUTPUT_DIR}/{symbol}_1m.parquet"
    
    # Save to parquet
    df.to_parquet(filename, compression='snappy')
    
    print(f"✓ Saved to {filename}")
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    
    return True


def main():
    """
    Main function to download and convert data for all symbols
    """
    print("="*70)
    print("BINANCE FUTURES DATA DOWNLOADER")
    print("Downloading 1-minute OHLCV data for October 2024")
    print("="*70)
    
    for symbol in SYMBOLS:
        try:
            # Download data
            klines = download_month_data(symbol, year=2024, month=10)
            
            if not klines:
                print(f"✗ Failed to download data for {symbol}")
                continue
            
            # Convert to DataFrame
            df = convert_to_dataframe(klines)
            
            if df.empty:
                print(f"✗ Failed to convert data for {symbol}")
                continue
            
            # Save to parquet
            success = save_to_parquet(df, symbol)
            
            if success:
                print(f"\n✓ Successfully processed {symbol}")
            else:
                print(f"\n✗ Failed to save {symbol}")
                
        except Exception as e:
            print(f"\n✗ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
