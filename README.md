# AI Trading Agent — Bybit

An AI-powered algorithmic trading system for cryptocurrency perpetual futures on **Bybit**. Signal-driven, risk-aware, backtestable.

> **Current status:** v2 — functional on testnet. Signal generator uses real technical indicators (not synthetic AI). All 7 risk management components are wired into the trade flow. Includes a historical backtester with standard metrics.

---

## Table of Contents

- [Architecture](#architecture)
- [Signal Generator](#signal-generator)
- [Risk Management](#risk-management)
- [Execution Strategies](#execution-strategies)
- [Backtesting](#backtesting)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development Status](#development-status)

---

## Architecture

```
                     ┌─────────────────────────────┐
                     │        TradingSystem         │
                     │      (main.py — loop)        │
                     └──────────┬──────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
   ┌──────────┐        ┌──────────────┐      ┌──────────────┐
   │  Market  │        │ Risk Gates   │      │  Execution   │
   │ Analysis │        │ (7 checks)   │      │  Strategies  │
   ├──────────┤        ├──────────────┤      ├──────────────┤
   │ OFI      │        │ Drawdown     │      │ HFT          │
   │ MarketInsights│   │ Max Loss     │      │ Market Maker │
   │ OrderBook│        │ Position Sizing│    │ Scalping     │
   │ Iceberg  │        │ Stop Loss/TP │      │ Default      │
   │ StopHunt │        │ Trailing Stop │     │              │
   │ Timing   │        │ Leverage     │      │              │
   └──────────┘        │ Volatility   │      └──────────────┘
                       └──────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │  SignalGenerator│
                     │  (technical +   │
                     │   regime-based) │
                     └─────────────────┘
```

### Data Flow

1. **Market Analysis** — Fetches OHLCV + order book from Bybit via unified `BybitClient` (ccxt + WebSocket)
2. **Signal Generation** — `SignalGenerator` computes RSI, MACD, Bollinger Bands, momentum, volume confirmation, and OFI; then fuses them into a combined signal with market regime detection
3. **Risk Gates** — All 7 risk components check the proposed trade before execution. Position size adapts to volatility, win rate, and consecutive losses
4. **Execution** — Order placed via `BybitClient`. Position monitored every cycle for SL/TP/trailing stop
5. **Feedback** — Trade outcomes recorded back into the `SignalGenerator` for adaptive weight adjustment

---

## Signal Generator

**File:** `ai/self_learning.py` — `SignalGenerator` class

Replaces the original synthetic DQN with a transparent, interpretable signal engine:

| Indicator | Weight | Description |
|-----------|--------|-------------|
| RSI (14) | 20% | Relative Strength Index — overbought/oversold oscillator |
| MACD | 20% | Moving Average Convergence Divergence histogram |
| Bollinger Bands %b | 15% | Position within volatility bands |
| Trend Strength | 15% | Linear regression slope normalized to [-1, 1] |
| Volume Confirmation | 10% | Rising volume on directional moves |
| Order Flow Imbalance | 10% | Buy vs sell pressure from order book |
| Momentum (ROC) | 10% | Rate of change over 10 periods |

### Regime Detection

Classifies each candle into one of four regimes:

- **trend_up** / **trend_down** — strong directional bias → amplify signal 20%
- **ranging** — no clear direction → neutral weighting
- **volatile** — high volatility → reduce confidence 50%

### Adaptive Behavior

- Tracks win rate, consecutive losses, and total trades
- Reduces position size by 50% after 3 consecutive losses
- Halts trading after 5 consecutive losses
- Indicator weights adjust based on historical performance

---

## Risk Management

All 7 risk components are initialized in `TradingSystem._init_risk()` and checked on every trade:

| Component | Gate | Behavior When Triggered |
|-----------|------|------------------------|
| `MaxDrawdown` | ≤15% (configurable) | Prevents new trades |
| `MaxLossPerTrade` | Per-trade loss limit | Prevents new trades |
| `RiskManager.check_risk()` | Volatility-adjusted | Blocks trade or reduces size |
| `RiskManager.check_volatility()` | High volatility flag | Reduces position size 50% |
| `StopLossTakeProfit` | 1.5% SL / 3% TP (configurable) | Closes position mid-cycle |
| `TrailingStopLoss` | Activates at 2% profit, 1% trail | Closes position mid-cycle |
| `PositionSizing` | Kelly-based fractional sizing | Calculates dynamic size per trade |

Position sizing also incorporates a **performance factor** from the signal generator — reduced after losses, full after wins.

---

## Execution Strategies

All strategies use **centralized data** from `BybitClient` — no duplicate WebSocket connections.

| Strategy | Trigger | Behavior |
|----------|---------|----------|
| **HFT** | Spread > 0.02% + strong order-book pressure | Single directional market order (buy or sell). Rate-limited to 1 per 5s. |
| **Market Maker** | Low-volatility regime | Places two-sided limit orders around mid-price. Cancels stale orders each cycle. Inventory-aware (stops if net position exceeds limit). |
| **Scalping** | Bid/ask volume ratio > 1.5 or < 0.67 | Single market order in direction of pressure. |
| **Default** | Any regime | Uses `AdvancedTradingStrategy` ensemble signal (Transformer + XGBoost/LightGBM — still under development). |

Strategy selection is automatic based on the current volatility regime:
- High volatility → HFT
- Strong OFI → Scalping
- Low volatility → Market Making
- Mixed → Default (combined signal)

---

## Backtesting

**File:** `backtest.py`

```
python backtest.py --symbol BTCUSDT --days 30
```

### Features
- Uses the **same `SignalGenerator`** as live trading — signals are identical
- Simulates fills at candle close with configurable slippage (default 2bps)
- Reports standard metrics:
  - Total return / Annualized return
  - Sharpe ratio (annualized)
  - Maximum drawdown
  - Win rate, profit factor, avg win/loss
- Saves trade log and equity curve to `backtest_results/` as CSV
- Can load historical data via API (mainnet) or from a CSV file

### Options

```
--symbol  BTCUSDT     Trading pair
--days    14          Days of hourly data to fetch
--balance 1000.0      Starting balance
--risk    0.01        Risk per trade (1%)
--file    data.csv    Optional: load CSV instead of API
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Bybit Testnet account ([sign up](https://testnet.bybit.com/))
- API keys with trade permissions

### Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/ai_trading_agent.git
cd ai_trading_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt python-dotenv

# 4. Configure API keys
cp .env.example .env
# Edit .env with your Bybit testnet API key/secret

# 5. Run a backtest first
python backtest.py --symbol BTCUSDT --days 14

# 6. Start live trading (testnet)
python main.py
```

---

## Configuration

All configuration is in **`config.py`** (auto-populated from environment variables with defaults).

Key settings (override via `.env` or system environment):

| Variable | Default | Description |
|----------|---------|-------------|
| `BYBIT_API_KEY` | — | Bybit API key |
| `BYBIT_API_SECRET` | — | Bybit API secret |
| `USE_TESTNET` | `True` | Testnet or mainnet |
| `SYMBOL_BTC` | `BTCUSDT` | Trading pair |
| `TRADE_SIZE_BTC` | `0.001` | Minimum trade size |
| `MAX_POSITION_BTC` | `0.1` | Maximum position |
| `INITIAL_BALANCE` | `1000.0` | Starting balance |
| `RISK_PER_TRADE` | `0.01` | 1% risk per trade |
| `MAX_DRAWDOWN` | `0.15` | 15% max drawdown |
| `MAX_LEVERAGE` | `3` | Maximum leverage |

See `config.py` for the full list of tunable parameters.

---

## Project Structure

```
├── main.py                  # Trading system orchestrator (entry point)
├── config.py                # Centralized configuration
├── backtest.py              # Historical backtester
├── .gitignore               # Standard Python gitignore
├── .env.example             # API key template
├── requirements.txt         # Python dependencies
│
├── bybit_client.py          # Unified API client (ccxt + WebSocket)
├── data_pipeline/           # Alternative client (legacy)
│   └── bybit_api.py
│
├── ai/
│   └── self_learning.py     # SignalGenerator: technical indicators + regime detection
│
├── analysis/
│   ├── market_analysis.py   # MarketInsights: OHLCV analysis, volatility, RSI
│   ├── ofi_analysis.py      # Order Flow Imbalance calculator
│   ├── order_book_analysis.py # Order book pressure indicators
│   ├── iceberg_detector.py  # Iceberg order detection
│   ├── stop_hunt_detector.py # Stop-hunt pattern detection
│   └── order_timing.py      # Large order detection
│
├── execution/
│   ├── hft_trading.py       # HFT strategy (spread + pressure-based)
│   ├── market_maker.py      # Market making (two-sided with inventory mgmt)
│   └── scalping_strategy.py # Scalping (pressure-based)
│
├── risk_management/
│   ├── leverage_control.py
│   ├── max_drawdown.py
│   ├── max_loss.py
│   ├── position_sizing.py
│   ├── risk_manager.py
│   ├── stop_loss_take_profit.py
│   └── trailing_stop.py
│
├── strategies/
│   ├── trading_strategy.py  # AdvancedTradingStrategy (Transformer + ensemble)
│   ├── buy_strategy.py
│   ├── sell_strategy.py
│   └── ...
│
├── tracking/
│   ├── profit_tracker.py
│   └── strategy_report.py
│
├── models/
│   └── order_book_lstm.py   # LSTM market predictor (legacy)
│
├── data/                    # Market data storage (gitignored)
├── saved_models/            # Trained models (gitignored)
├── reports/                 # Generated reports (gitignored)
├── backtest_results/        # Backtest output (gitignored)
│
└── keys/                    # API key files (gitignored)
```

---

## Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| API Client (BybitClient) | ✅ Stable | ccxt + WebSocket, retry logic, failover |
| Signal Generator | ✅ Stable | 6 technical indicators, regime detection, adaptive weights |
| Risk Management | ✅ Stable | All 7 components wired and checked on every trade |
| Market Analysis | ✅ Stable | OFI, order book, market insights, stop-hunt, timing |
| Execution (HFT) | ✅ Functional | Directional, rate-limited, centralized data |
| Execution (Market Making) | ✅ Functional | Order tracking, cancellation, inventory mgmt |
| Execution (Scalping) | ✅ Functional | Pressure-based, centralized data |
| Backtesting | ✅ Built | Same signal generator, standard metrics, CSV output |
| AdvancedTradingStrategy | 🔧 Partial | Transformer/ensemble architecture exists but needs training data pipeline |
| Unit Tests | 🔧 Missing | Needed for regression safety |
| Mainnet Trading | ❌ Not ready | Requires wallet-level risk and multi-collateral support |

### Known Limitations

- The `AdvancedTradingStrategy` (Transformer + XGBoost/LightGBM ensemble) has its model infrastructure set up but lacks a connected training data pipeline — its signal weight is currently null
- Market maker's `cancel_order` depends on a method that may need verification against the live Bybit API
- No GPU acceleration for model training (intentional — CPU inference is sufficient for the signal generator)
- The `data_pipeline/` directory still contains the legacy `BybitAPI` class (pybit-based) — kept for reference but not used by the main system

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Collins Oloo — [collaustine27@gmail.com](mailto:collaustine27@gmail.com)