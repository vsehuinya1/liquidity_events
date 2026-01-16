# Liquidity Events Trading Bot

A real-time cryptocurrency trading bot that detects liquidity events and executes paper trades with Telegram notifications.

## Features

- **WebSocket Feed Handler**: Real-time data streaming with 5-minute resampling
- **Live Event Detection**: Identifies sweeps, clusters, and thinning patterns
- **Telegram Integration**: Instant alerts for entries, exits, and daily summaries
- **Paper Trading**: Safe testing environment with daily limits
- **Thread-Safe Architecture**: Production-ready design

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the example config:
```bash
cp config.example.py config.py
```

3. Edit `config.py` with your Telegram bot token and chat ID

4. Run the bot:
```bash
python main.py
```

## Configuration

See `config.example.py` for all available configuration options.

## Paper Trading

The bot includes a paper trading executor that simulates trades without real capital. Check `logs/paper_trading_*.log` for trade history.

## Version History

- v0.4.0: Paper trading with Telegram alerts
- v1.1.0: Multi-pair support with global risk containment and improved WebSocket handler
