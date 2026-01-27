# Liquidity Events Trading Bot (Version F - VPS Edition)

A real-time cryptocurrency trading bot that detects liquidity events (sweeps, clusters) using forensic 1-minute logic and executes trades with risk containment.

## Version F Features (VPS Verified)

- **1-Minute Forensic Logic**: Detects `Cluster` and `Sweep` events using 1-minute bars (previously 5m) to match backtest sensitivity.
- **Persistent State**: Cluster state and trade history are saved to disk (`data/state/strategy_state.json`), surviving restarts.
- **WebSocket Feed**: Aggregates 1m bars from Binance Futures and streams them directly to the strategy.
- **Risk Engine**: Daily loss limits (-100R testnet) and correlation guards.
- **Telegram Command & Control**: Real-time alerts and `/KILL` switch functionality.

## Quick Start (VPS)

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/vsehuinya1/liquidity_events_version_F.git
   cd liquidity_events_version_F
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Copy `config.example.py` to `config.py` and set your `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and `BINANCE_API_KEY`.

3. **Run**:
   ```bash
   # Use PM2 for production
   pm2 start main_testnet.py --name "trading_bot_vF" --interpreter python3
   ```

   **Monitor Logs**:
   ```bash
   pm2 logs trading_bot_vF
   ```

## Backtest Comparison

| Feature | Backtest (Option F) | Live Bot (Version F) |
| :--- | :--- | :--- |
| **Timeframe** | 1-minute | **1-minute** (Synced ✅) |
| **Lookback** | 20 bars (20m) | **20 bars** (20m) (Synced ✅) |
| **Logic** | Dynamic Filters | **Dynamic Filters** (Synced ✅) |
| **Cooldown** | ~1 Hour ATR | **Disabled** (By Request) |

## Repository Structure

- `main_testnet.py`: Entry point, orchestrates feed, risk, and execution.
- `live_event_detector_gem.py`: **Core Logic**. Handles signal detection and state.
- `websocket_handler_improved.py`: Manages Binance connection and bar buffering.
- `strategy_orchestrator.py`: Routes signals and manages fleet state.
