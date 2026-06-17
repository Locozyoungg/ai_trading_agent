"""
Centralized configuration for the AI Trading Agent.
Loads from environment variables with sensible defaults.
All hardcoded paths are eliminated — use these values everywhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
BACKTEST_RESULTS_DIR = PROJECT_ROOT / "backtest_results"

for d in [DATA_DIR, REPORTS_DIR, SAVED_MODELS_DIR, BACKTEST_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Credentials ────────────────────────────────────────────────────────
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# ── Trading Parameters ─────────────────────────────────────────────────────
SYMBOL = os.getenv("SYMBOL_BTC", "BTCUSDT")
TRADE_SIZE = float(os.getenv("TRADE_SIZE_BTC", "0.001"))
MAX_POSITION = float(os.getenv("MAX_POSITION_BTC", "0.1"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "1000.0"))

# ── Risk Parameters ────────────────────────────────────────────────────────
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))       # 1% per trade
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.15"))            # 15% max drawdown
STOP_LOSS_FACTOR = 0.015                                           # 1.5% stop-loss
TAKE_PROFIT_FACTOR = 0.03                                          # 3% take-profit
TRAILING_STOP_ACTIVATE = 0.02                                      # 2% profit → activate trailing
TRAILING_STOP_DISTANCE = 0.01                                      # 1% trailing distance
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "3"))                 # 3x max leverage
VOLATILITY_THRESHOLD = 0.3                                         # high-volatility threshold

# ── Strategy Parameters ────────────────────────────────────────────────────
LOOKBACK_PERIOD = 60               # candles for indicator calculation
VOLATILITY_WINDOW = 20             # candles for volatility estimation
SEQUENCE_LENGTH = 50               # LSTM sequence length (if used)
TRANSFORMER_WEIGHT = 0.35          # signal weight — transformer model
ENSEMBLE_WEIGHT = 0.35             # signal weight — ensemble model
MOMENTUM_WEIGHT = 0.15             # signal weight — momentum
VOLATILITY_WEIGHT = 0.15           # signal weight — vol regime
SIGNAL_THRESHOLD = 0.3             # minimum combined signal to trade

# ── Analysis Parameters ────────────────────────────────────────────────────
OFI_LEVELS = 5                     # order-book levels for OFI calculation
ICEBERG_CYCLES = 10                # poll cycles for iceberg detection
ICEBERG_REFRESH_THRESHOLD = 3      # size changes to flag as iceberg
STOP_HUNT_THRESHOLD = 0.005        # 0.5% price move → possible stop hunt
STOP_HUNT_LOOKBACK = 60            # seconds for stop-hunt window

# ── Execution Parameters ───────────────────────────────────────────────────
TRADE_LOOP_INTERVAL = 2            # seconds between trade cycles
REPORT_INTERVAL = 3600             # seconds between strategy reports
POSITION_SYNC_INTERVAL = 60        # seconds between position re-syncs
HFT_SPREAD_THRESHOLD = 0.0002      # 0.02% minimum spread for HFT
HFT_ORDER_SIZE = 0.001             # BTC per HFT leg
MARKET_MAKER_SPREAD = 0.0005       # 0.05% spread for market making
MARKET_MAKER_SIZE = 0.01           # BTC per market-making leg
SCALPING_SPREAD = 0.0002           # 0.02% scalping target
SCALPING_SIZE = 0.001              # BTC per scalp

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "trading_bot.log"
LOG_LEVEL = "DEBUG"
CONSOLE_LOG_LEVEL = "INFO"

# ── Model Paths ────────────────────────────────────────────────────────────
ORDER_BOOK_MODEL_PATH = str(SAVED_MODELS_DIR / "order_book_predictor.h5")
TRADING_MODEL_PATH = str(SAVED_MODELS_DIR / "advanced_trading_model.keras")
DATA_FILE = str(DATA_DIR / f"combined_market_{SYMBOL}.csv")

# ── Validation ─────────────────────────────────────────────────────────────
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    import logging
    logging.warning("BYBIT_API_KEY or BYBIT_API_SECRET not set in environment or .env file")