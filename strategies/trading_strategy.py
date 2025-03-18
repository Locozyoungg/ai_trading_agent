# strategies/trading_strategy.py
import time
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from bybit_client import BybitClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    MODEL_SAVE_DIR = Path("saved_models")
    RETRAIN_INTERVAL_HOURS = 12  # Faster retraining for adaptability
    MAX_API_RETRIES = 3
    RETRY_DELAY = 5
    MIN_TRAINING_SAMPLES = 1000  # More data for robust training
    MODEL_SAVE_FORMAT = "model_{timestamp}.h5"
    SCALER_SAVE_FORMAT = "scaler_{timestamp}.pkl"

class AdvancedTradingStrategy:
    def __init__(
        self,
        client: BybitClient,
        symbol: str = "BTCUSDT",
        N: int = 10,  # Your original order book depth
        initial_threshold: float = 0.2,  # Your original threshold
        interval: int = 10,  # Your original cycle time
        lookback_period: int = 20,  # Your original lookback
        volatility_window: int = 10,  # Your original volatility window
        risk_per_trade: float = 0.01,  # Your original risk
        stop_loss_factor: float = 0.02,  # Your original SL
        take_profit_factor: float = 0.04,  # Your original TP
        lstm_sequence_length: int = 60  # Your original LSTM sequence
    ):
        self.client = client
        self.symbol = symbol
        self.N = N
        self.threshold = initial_threshold
        self.interval = interval
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.risk_per_trade = risk_per_trade
        self.stop_loss_factor = stop_loss_factor
        self.take_profit_factor = take_profit_factor
        self.lstm_sequence_length = lstm_sequence_length

        # Enhanced features (Jim Simons-inspired)
        self.momentum_window_short = 5  # Short-term MA for momentum
        self.momentum_window_long = 50  # Long-term MA for momentum
        self.imbalance_momentum_weight = 0.5  # Blend order book with momentum
        self.stat_arb_factor = 0.1  # Statistical arbitrage adjustment

        self.position = None
        self.entry_price = None
        self.position_size = None
        self.lstm_model = None
        self.scaler = None
        self.last_trained = None

        Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)
        self._load_latest_model()
        logger.info("Advanced Trading Strategy initialized with Jim Simons-inspired enhancements")

    def _safe_api_call(self, api_method, *args, **kwargs):
        for attempt in range(Config.MAX_API_RETRIES):
            try:
                return api_method(*args, **kwargs)
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt+1}): {str(e)}")
                if attempt < Config.MAX_API_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
        raise Exception(f"API method {api_method.__name__} failed after {Config.MAX_API_RETRIES} attempts")

    def compute_imbalance(self, orderbook: Dict) -> float:
        """Your original order book imbalance calculation."""
        try:
            bids = [(float(p), float(s)) for p, s in orderbook.get('b', [])]
            asks = [(float(p), float(s)) for p, s in orderbook.get('a', [])]
            top_bids = sorted(bids, key=lambda x: x[0], reverse=True)[:self.N]
            top_asks = sorted(asks, key=lambda x: x[0])[:self.N]
            total_buy = sum(s for _, s in top_bids)
            total_sell = sum(s for _, s in top_asks)
            total = total_buy + total_sell
            return (total_buy - total_sell) / total if total > 0 else 0
        except Exception as e:
            logger.error(f"Order book processing error: {str(e)}")
            return 0

    def get_current_price(self) -> Optional[float]:
        return self._safe_api_call(self.client.get_current_price, self.symbol)

    def get_historical_data(self, limit: int) -> Optional[np.ndarray]:
        data = self._safe_api_call(self.client.get_historical_data, self.symbol, limit=limit)
        return np.array([float(c[4]) for c in data]) if data else None

    def calculate_volatility(self, prices: np.ndarray) -> Optional[float]:
        """Your original volatility calculation."""
        if len(prices) < 2:
            return None
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-self.volatility_window:])

    def calculate_momentum(self, prices: np.ndarray) -> float:
        """Jim Simons-inspired momentum using short/long MA crossover."""
        if len(prices) < self.momentum_window_long:
            return 0.0
        short_ma = np.mean(prices[-self.momentum_window_short:])
        long_ma = np.mean(prices[-self.momentum_window_long:])
        return (short_ma - long_ma) / long_ma  # Normalized momentum

    def compute_stat_arb_signal(self, orderbook: Dict, prices: np.ndarray) -> float:
        """Statistical arbitrage signal inspired by Simons’ quantitative approach."""
        if len(prices) < self.lookback_period:
            return 0.0
        bid_ask_spread = float(orderbook.get('a', [[0]])[0][0]) - float(orderbook.get('b', [[0]])[0][0])
        mean_spread = np.mean([float(a[0]) - float(b[0]) for a, b in zip(orderbook.get('a', [])[:self.N], orderbook.get('b', [])[:self.N])])
        z_score = (bid_ask_spread - mean_spread) / (self.calculate_volatility(prices) or 0.01)
        return self.stat_arb_factor * z_score  # Small influence to enhance signal

    def _train_lstm_model_impl(self, data: np.ndarray) -> Tuple[Sequential, MinMaxScaler]:
        """Your original LSTM training with slight enhancements."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.lstm_sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.lstm_sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([
            LSTM(75, return_sequences=True, input_shape=(X.shape[1], 1)),  # Slightly deeper
            Dropout(0.2),  # Regularization
            LSTM(50),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=15, batch_size=32, verbose=0)  # More epochs for precision
        return model, scaler

    def _save_model(self, model: Sequential, scaler: MinMaxScaler):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Config.MODEL_SAVE_DIR / Config.MODEL_SAVE_FORMAT.format(timestamp=timestamp)
        scaler_path = Config.MODEL_SAVE_DIR / Config.SCALER_SAVE_FORMAT.format(timestamp=timestamp)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved model: {model_path}, scaler: {scaler_path}")

    def _load_latest_model(self):
        model_files = sorted(Config.MODEL_SAVE_DIR.glob("model_*.h5"), reverse=True)
        scaler_files = sorted(Config.MODEL_SAVE_DIR.glob("scaler_*.pkl"), reverse=True)
        if model_files and scaler_files:
            self.lstm_model = load_model(model_files[0])
            self.scaler = joblib.load(scaler_files[0])
            self.last_trained = datetime.fromtimestamp(model_files[0].stat().st_mtime)
            logger.info(f"Loaded existing model: {model_files[0]}")

    def _retrain_model_if_needed(self):
        if not self.last_trained or (datetime.now() - self.last_trained) > timedelta(hours=Config.RETRAIN_INTERVAL_HOURS):
            logger.info("Initiating model retraining...")
            self.train_lstm_model()

    def train_lstm_model(self):
        hist_data = self.get_historical_data(self.lookback_period + self.volatility_window + self.lstm_sequence_length)
        if hist_data is None or len(hist_data) < Config.MIN_TRAINING_SAMPLES:
            logger.error("Insufficient data for training")
            return
        model, scaler = self._train_lstm_model_impl(hist_data)
        self.lstm_model = model
        self.scaler = scaler
        self.last_trained = datetime.now()
        self._save_model(model, scaler)

    def predict_price_movement(self, data: np.ndarray) -> Optional[float]:
        """Your original LSTM prediction."""
        if self.lstm_model is None or self.scaler is None:
            return None
        if len(data) < self.lstm_sequence_length:
            return None
        last_sequence = data[-self.lstm_sequence_length:]
        scaled_seq = self.scaler.transform(last_sequence.reshape(-1, 1))
        scaled_seq = scaled_seq.reshape(1, self.lstm_sequence_length, 1)
        predicted = self.lstm_model.predict(scaled_seq, verbose=0)
        return self.scaler.inverse_transform(predicted)[0, 0] - data[-1]

    def get_signal(self) -> float:
        """Blended signal: Your order book + Simons’ momentum and stat arb."""
        orderbook = self._safe_api_call(self.client.get_order_book, self.symbol)
        hist_data = self.get_historical_data(self.lookback_period + self.volatility_window + self.lstm_sequence_length)
        if not orderbook or hist_data is None:
            return 0.0

        # Your original components
        imbalance = self.compute_imbalance(orderbook)
        volatility = self.calculate_volatility(hist_data)
        predicted_change = self.predict_price_movement(hist_data) or 0.0

        # Jim Simons-inspired components
        momentum = self.calculate_momentum(hist_data)
        stat_arb = self.compute_stat_arb_signal(orderbook, hist_data)

        # Blend signals (weighted combination)
        signal = (
            0.5 * imbalance +  # Your order book core
            0.2 * (predicted_change / hist_data[-1] if hist_data[-1] != 0 else 0) +  # LSTM confirmation
            0.2 * momentum +  # Simons’ momentum
            0.1 * stat_arb  # Simons’ statistical arbitrage
        )
        self.adjust_threshold(volatility)
        return 1.0 if signal > self.threshold else -1.0 if signal < -self.threshold else 0.0

    def adjust_threshold(self, volatility: float):
        """Your original threshold adjustment with a Simons twist."""
        if volatility is not None:
            self.threshold = max(self.initial_threshold, volatility * 1.5 + abs(self.stat_arb_factor))  # Dynamic with stat arb influence

    def calculate_position_size(self, balance: float, price: float, volatility: float) -> float:
        """Your original sizing with volatility adjustment."""
        risk_adjusted = self.risk_per_trade / (volatility if volatility else 0.01)
        return min((balance * risk_adjusted) / price, 0.1)  # Cap at 0.1 BTC

    def execute_trade(self):
        """Execution logic compatible with main.py, blending both strategies."""
        signal = self.get_signal()
        current_price = self.get_current_price()
        balance = self.client.get_balance()
        if not all([current_price, balance]):
            return
        hist_data = self.get_historical_data(self.volatility_window)
        volatility = self.calculate_volatility(hist_data) or 0.01
        size = self.calculate_position_size(balance, current_price, volatility)

        if signal > 0 and not self.position:
            self.place_order("Buy", size)
            self.position = 'long'
            self.entry_price = current_price
            self.position_size = size
            logger.info(f"Buy executed: {size} @ {current_price}, signal={signal}")
        elif signal < 0 and not self.position:
            self.place_order("Sell", size)
            self.position = 'short'
            self.entry_price = current_price
            self.position_size = size
            logger.info(f"Sell executed: {size} @ {current_price}, signal={signal}")
        elif self.position:
            self.manage_position(current_price)

    def place_order(self, side: str, qty: float):
        self._safe_api_call(self.client.place_order, self.symbol, qty=qty, side=side, order_type="Market")
        logger.info(f"Order executed: {side} {qty} {self.symbol}")

    def manage_position(self, current_price: float):
        """Your original position management."""
        if self.position == 'long':
            if current_price <= self.entry_price * (1 - self.stop_loss_factor) or \
               current_price >= self.entry_price * (1 + self.take_profit_factor):
                self.place_order("Sell", self.position_size)
                logger.info(f"Closed long position at {current_price}")
                self.position = None
        elif self.position == 'short':
            if current_price >= self.entry_price * (1 + self.stop_loss_factor) or \
               current_price <= self.entry_price * (1 - self.take_profit_factor):
                self.place_order("Buy", self.position_size)
                logger.info(f"Closed short position at {current_price}")
                self.position = None

    def run(self):
        if not self.lstm_model:
            self.train_lstm_model()
        while True:
            self._retrain_model_if_needed()
            self.execute_trade()
            time.sleep(self.interval)

if __name__ == "__main__":
    client = BybitClient("YOUR_API_KEY", "YOUR_API_SECRET", testnet=True)
    strategy = AdvancedTradingStrategy(client, symbol="BTCUSDT")
    strategy.run()
