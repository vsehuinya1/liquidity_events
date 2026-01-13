#!/usr/bin/env python3
"""
Run backtest on SOLUSDT December 2025 data
"""

import sys
sys.path.append('/Users/MartinOile/Desktop/liquidity_events')

# Change DATA_PATH to SOLUSDT
DATA_PATH = 'data/parquet/SOLUSDT_1m.parquet'

# Now run the strategy
exec(open('/Users/MartinOile/Desktop/liquidity_events/master_thinning_strategy.py').read())
