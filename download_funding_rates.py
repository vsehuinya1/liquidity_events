#!/usr/bin/env python3
"""
Download Binance Futures Funding Rate History for SOLUSDT (Dec 2025)
"""
import requests
import pandas as pd
import time
from datetime import datetime
import os

BASE_URL = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/fundingRate"
SYMBOL = "SOLUSDT"
START_DATE = datetime(2025, 12, 1)
# End date is strictly end of Dec 2025
END_DATE = datetime(2026, 1, 1) 
OUTPUT_FILE = "data/parquet/SOLUSDT_funding_dec2025.parquet"
os.makedirs("data/parquet", exist_ok=True)

def get_funding_rates(symbol, start_time, end_time, limit=1000):
    all_rates = []
    current_start = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": limit
        }
        
        try:
            resp = requests.get(BASE_URL + ENDPOINT, params=params)
            data = resp.json()
            
            if not data:
                break
                
            all_rates.extend(data)
            last_entry = data[-1]
            current_start = last_entry['fundingTime'] + 1000 # Advance 1s
            
            print(f"Fetched {len(data)} rates, up to {datetime.fromtimestamp(last_entry['fundingTime']/1000)}")
            time.sleep(0.1) # Rate limit safety
            
        except Exception as e:
            print(f"Error: {e}")
            break
            
    return all_rates

print(f"Downloading funding rates for {SYMBOL} from {START_DATE} to {END_DATE}...")
rates = get_funding_rates(SYMBOL, START_DATE, END_DATE)

if rates:
    df = pd.DataFrame(rates)
    # fundingTime, fundingRate, symbol
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df.set_index('fundingTime', inplace=True)
    df.index.name = 'timestamp'
    df = df[['fundingRate']] # Keep only rate
    
    # Save
    df.to_parquet(OUTPUT_FILE)
    print(f"Saved {len(df)} records to {OUTPUT_FILE}")
else:
    print("No data found.")
