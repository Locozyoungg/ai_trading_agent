# strategies/trading_strategy.py
"""
Advanced Trading Strategy with Transformer and Enhanced Ensemble Models
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    joblib = None
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from bybit_client import BybitClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('trading_strategy.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [file_handler, stream_handler]

class Config:
    MODEL_SAVE_DIR = Path.home() / "OneDrive" / "ai_trading_agent" / "saved_models"
    RETRAIN_INTERVAL_HOURS = 12
    MAX_API_RETRIES = 3
    RETRY_DELAY = 5
    MIN_TRAINING_SAMPLES = 1000
    MODEL_SAVE_FORMAT = "transformer_model_{timestamp}.h5"
    SCALER_SAVE_FORMAT = "scaler_{timestamp}.pkl"
    ENSEMBLE_MODEL_PATH = MODEL_SAVE_DIR / "ensemble_models.pkl"

class AdvancedTradingStrategy:
    def __init__(
        self,
        client: BybitClient,
        symbol: str = "BTCUSDT",
        N: int = 10,
        initial_threshold: float = 0.2,
        interval: int = 5,
        lookback_period: int = 60,
        volatility_window: int = 20,
        risk_per_trade: float = 0.02,
        stop_loss_factor: float = 0.015,
        take_profit_factor: float = 0.03,
        sequence_length: int = 60,
        ensemble_confidence_threshold: float = 0.65
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
        self.sequence_length = sequence_length
        self.ensemble_confidence_threshold = ensemble_confidence_threshold

        self.position = None
        self.entry_price = None
        self.position_size = None
        self.signal_weights = {'transformer': 0.35, 'ensemble': 0.35, 'momentum': 0.15, 'volatility': 0.15}
        self.performance_history = []
        
        self.transformer_model = None
        self.scaler = RobustScaler()
        self.last_trained = None
        self.ensemble_models = {
            'gb': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'xgb': XGBClassifier(n_estimators=200, random_state=42, tree_method='hist'),
            'lgb': LGBMClassifier(n_estimators=200, random_state=42, device='cpu')  # Adjust to 'gpu' if available
        }

        Config.MODEL_SAVE_DIR.mkdir(exist_ok=True)
        self._load_latest_models()
        logger.info("AdvancedTradingStrategy initialized with Transformer and ensemble capabilities")

    def _safe_api_call(self, api_method, *args, **kwargs) -> Optional[any]:
        for attempt in range(Config.MAX_API_RETRIES):
            try:
                return api_method(*args, **kwargs)
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt+1}): {str(e)}")
                if attempt < Config.MAX_API_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
        logger.error(f"API method {api_method.__name__} failed after {Config.MAX_API_RETRIES} attempts")
        return None

    def get_multi_timeframe_data(self, limits: Dict[str, int]) -> Dict[str, Optional[np.ndarray]]:
        timeframes = {'1m': limits.get('1m', self.lookback_period),
                      '5m': limits.get('5m', self.lookback_period // 5),
                      '15m': limits.get('15m', self.lookback_period // 15)}
        data = {}
        for tf, limit in timeframes.items():
            ohlcv = self._safe_api_call(self.client.get_historical_data, self.symbol, interval=tf, limit=limit)
            data[tf] = ohlcv if ohlcv is not None else None
        return data

    def get_current_price(self) -> Optional[float]:
        return self._safe_api_call(self.client.get_current_price, self.symbol)

    def compute_imbalance(self, orderbook: Dict) -> float:
        try:
            bids = [(float(p), float(s)) for p, s in orderbook['bids']][:self.N]
            asks = [(float(p), float(s)) for p, s in orderbook['asks']][:self.N]
            bid_volume = sum(s for _, s in bids)
            ask_volume = sum(s for _, s in asks)
            total_volume = bid_volume + ask_volume
            return (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        except Exception as e:
            logger.error(f"Order book imbalance error: {str(e)}")
            return 0.0

    def calculate_volatility(self, hist_data: np.ndarray) -> Optional[float]:
        if hist_data is None or len(hist_data) < 2:
            return None
        returns = np.diff(hist_data[:, 4]) / hist_data[:-1, 4]
        return np.std(returns[-self.volatility_window:]) * np.sqrt(365 * 24 * 60)

    def calculate_momentum(self, hist_data: Dict[str, np.ndarray]) -> float:
        momentum = 0
        for tf, data in hist_data.items():
            if data is not None and len(data) > 1:
                momentum += (data[-1, 4] - data[0, 4]) / data[0, 4] if data[0, 4] != 0 else 0
        return momentum / len(hist_data)

    def _feature_engineering(self, hist_data: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(hist_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Technical Indicators
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Fix ATR calculation
        df['close_prev'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close_prev']).abs()
        df['tr3'] = (df['low'] - df['close_prev']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        df.drop(['close_prev', 'tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

        # Advanced Features
        df['hurst'] = df['close'].rolling(20).apply(self._compute_hurst, raw=True)
        df['entropy'] = df['close'].rolling(20).apply(self._shannon_entropy, raw=True)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        df.dropna(inplace=True)
        return df

    def _compute_hurst(self, series):
        lags = range(2, 20)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    def _shannon_entropy(self, series):
        p = np.histogram(series, bins=20, density=True)[0]
        return -np.sum(p * np.log2(p + 1e-10))

    def _build_transformer_model(self) -> Model:
        inputs = Input(shape=(self.sequence_length, 10))  # 10 features
        x = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation='tanh')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001), loss='mse', metrics=['mae'])
        return model

    def _train_ensemble_models(self, df: pd.DataFrame):
        features = ['sma_10', 'sma_50', 'rsi_14', 'macd', 'hurst', 'entropy', 'vwap', 'atr']
        X = self.scaler.fit_transform(df[features])
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        X = X[:-1]
        y = df['target'][:-1]

        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, _ in tscv.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            for name, model in self.ensemble_models.items():
                model.fit(X_train, y_train)

        joblib.dump(self.ensemble_models, Config.ENSEMBLE_MODEL_PATH)
        logger.info("Ensemble models trained and saved")

    def _train_transformer_model(self, df: pd.DataFrame):
        features = ['sma_10', 'sma_50', 'rsi_14', 'macd', 'hurst', 'entropy', 'vwap', 'atr', 'close', 'volume']
        X = self.scaler.fit_transform(df[features])
        X_seq = np.array([X[i:i+self.sequence_length] for i in range(len(X) - self.sequence_length)])
        y = (df['close'].shift(-1) > df['close']).astype(int)[self.sequence_length:].values
        self.transformer_model.fit(X_seq, y, epochs=15, batch_size=32, verbose=0)
        logger.info("Transformer model trained")

    def _predict_ensemble(self, df: pd.DataFrame) -> float:
        features = ['sma_10', 'sma_50', 'rsi_14', 'macd', 'hurst', 'entropy', 'vwap', 'atr']
        X = self.scaler.transform(df[features].iloc[-1:].values)
        probs = [model.predict_proba(X)[:, 1][0] for model in self.ensemble_models.values()]
        avg_prob = np.mean(probs)
        return 1.0 if avg_prob > self.ensemble_confidence_threshold else -1.0 if avg_prob < (1 - self.ensemble_confidence_threshold) else 0.0

    def _predict_transformer(self, df: pd.DataFrame) -> float:
        if not self.transformer_model:
            return 0.0
        features = ['sma_10', 'sma_50', 'rsi_14', 'macd', 'hurst', 'entropy', 'vwap', 'atr', 'close', 'volume']
        X = self.scaler.transform(df[features].tail(self.sequence_length))
        X_seq = X.reshape(1, self.sequence_length, -1)
        return self.transformer_model.predict(X_seq, verbose=0)[0][0]

    def _load_latest_models(self):
        try:
            model_files = sorted(Config.MODEL_SAVE_DIR.glob("transformer_model_*.h5"), reverse=True)
            scaler_files = sorted(Config.MODEL_SAVE_DIR.glob("scaler_*.pkl"), reverse=True)
            if model_files and scaler_files:
                self.transformer_model = self._build_transformer_model()
                self.transformer_model.load_weights(model_files[0])
                self.scaler = joblib.load(scaler_files[0])
                self.last_trained = datetime.fromtimestamp(model_files[0].stat().st_mtime)
                logger.info(f"Loaded Transformer model: {model_files[0]}")
            if Config.ENSEMBLE_MODEL_PATH.exists():
                self.ensemble_models = joblib.load(Config.ENSEMBLE_MODEL_PATH)
                logger.info("Loaded ensemble models")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            self.transformer_model = self._build_transformer_model()

    def _save_models(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Config.MODEL_SAVE_DIR / Config.MODEL_SAVE_FORMAT.format(timestamp=timestamp)
        scaler_path = Config.MODEL_SAVE_DIR / Config.SCALER_SAVE_FORMAT.format(timestamp=timestamp)
        self.transformer_model.save_weights(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.ensemble_models, Config.ENSEMBLE_MODEL_PATH)
        logger.info(f"Saved Transformer: {model_path}, Scaler: {scaler_path}, Ensemble: {Config.ENSEMBLE_MODEL_PATH}")

    def _retrain_model_if_needed(self):
        if not self.last_trained or (datetime.now() - self.last_trained) > timedelta(hours=Config.RETRAIN_INTERVAL_HOURS):
            hist_data = self.get_multi_timeframe_data({'1m': Config.MIN_TRAINING_SAMPLES + self.sequence_length})['1m']
            if hist_data is None or len(hist_data) < Config.MIN_TRAINING_SAMPLES:
                logger.error("Insufficient data for retraining")
                return
            df = self._feature_engineering(hist_data)
            self._train_ensemble_models(df)
            self._train_transformer_model(df)
            self.last_trained = datetime.now()
            self._save_models()
            logger.info("Models retrained successfully")

    def adjust_signal_weights(self, volatility: Optional[float], success_rate: float):
        if volatility and success_rate:
            self.signal_weights['transformer'] = min(0.45, max(0.25, self.signal_weights['transformer'] * (1 + success_rate)))
            self.signal_weights['ensemble'] = min(0.4, max(0.2, self.signal_weights['ensemble'] * (1 + success_rate * volatility)))
            self.signal_weights['momentum'] = min(0.2, max(0.05, self.signal_weights['momentum'] * (1 + success_rate)))
            self.signal_weights['volatility'] = min(0.2, max(0.05, self.signal_weights['volatility'] * (1 + volatility)))
            total = sum(self.signal_weights.values())
            for key in self.signal_weights:
                self.signal_weights[key] /= total
            logger.debug(f"Adjusted signal weights: {self.signal_weights}")

    def get_signal(self) -> float:
        orderbook = self._safe_api_call(self.client.get_order_book, self.symbol)
        hist_data_multi = self.get_multi_timeframe_data({'1m': self.lookback_period + self.sequence_length})
        hist_data = hist_data_multi['1m']
        if not orderbook or not hist_data:
            logger.warning("Missing data, signal=0")
            return 0.0

        self._retrain_model_if_needed()
        df = self._feature_engineering(hist_data)
        if len(df) < self.sequence_length:
            return 0.0

        transformer_signal = self._predict_transformer(df)
        ensemble_signal = self._predict_ensemble(df)
        momentum = self.calculate_momentum(hist_data_multi)
        volatility = self.calculate_volatility(hist_data)

        signal = (
            self.signal_weights['transformer'] * transformer_signal +
            self.signal_weights['ensemble'] * ensemble_signal +
            self.signal_weights['momentum'] * (1.0 if momentum > 0 else -1.0 if momentum < 0 else 0.0) +
            self.signal_weights['volatility'] * np.tanh(volatility if volatility else 0)
        )
        logger.debug(f"Signal components: transformer={transformer_signal}, ensemble={ensemble_signal}, momentum={momentum}, volatility={volatility}, total={signal}")
        
        success_rate = self._calculate_signal_success()
        self.adjust_signal_weights(volatility, success_rate)
        self.adjust_threshold(volatility)
        return 1.0 if signal > self.threshold else -1.0 if signal < -self.threshold else 0.0

    def _calculate_signal_success(self) -> float:
        if not self.performance_history:
            return 0.5
        successes = sum(1 for profit in self.performance_history[-20:] if profit > 0)
        return successes / min(len(self.performance_history), 20)

    def adjust_threshold(self, volatility: Optional[float]):
        if volatility:
            self.threshold = max(0.1, min(0.5, volatility * 1.5 + 0.05 * (1 - self._calculate_signal_success())))
            logger.debug(f"Threshold adjusted to {self.threshold}")

    def calculate_position_size(self, balance: float, price: float, volatility: float) -> float:
        win_prob = self._calculate_signal_success() or 0.5
        odds = self.take_profit_factor / self.stop_loss_factor
        kelly_fraction = max(0.01, min(0.25, (win_prob * (odds + 1) - 1) / odds if odds > 0 else self.risk_per_trade))
        size = (balance * kelly_fraction) / (price * max(volatility, 0.01))
        return min(size, balance / price)

    def place_order(self, side: str, qty: float):
        for attempt in range(Config.MAX_API_RETRIES):
            try:
                order = self.client.place_order(self.symbol, qty, side.upper(), order_type="Market")
                logger.info(f"Order executed: {side} {qty} {self.symbol}")
                return order
            except Exception as e:
                logger.error(f"Order failed (attempt {attempt+1}): {str(e)}")
                if attempt < Config.MAX_API_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
        logger.error("Order placement failed after retries")
        raise Exception("Order execution failed")

    def manage_position(self, current_price: float):
        if not self.position:
            return
        profit = (current_price - self.entry_price) / self.entry_price if self.position == 'long' else (self.entry_price - current_price) / self.entry_price
        self.performance_history.append(profit)
        if self.position == 'long':
            if profit >= self.take_profit_factor:
                self.place_order("SELL", self.position_size)
                logger.info(f"Closed long at {current_price} with profit {profit*100:.2f}%")
                self.position = None
            elif profit <= -self.stop_loss_factor:
                self.place_order("SELL", self.position_size)
                logger.info(f"Stopped out long at {current_price} with loss {profit*100:.2f}%")
                self.position = None
        elif self.position == 'short':
            if profit >= self.take_profit_factor:
                self.place_order("BUY", self.position_size)
                logger.info(f"Closed short at {current_price} with profit {profit*100:.2f}%")
                self.position = None
            elif profit <= -self.stop_loss_factor:
                self.place_order("BUY", self.position_size)
                logger.info(f"Stopped out short at {current_price} with loss {profit*100:.2f}%")
                self.position = None

    def execute_trade(self):
        signal = self.get_signal()
        current_price = self.get_current_price()
        balance = self._safe_api_call(self.client.get_balance) or 0
        logger.debug(f"Signal: {signal}, Price: {current_price}, Balance: {balance}")
        if not all([current_price, balance]):
            logger.warning("Missing data, skipping trade")
            return
        hist_data = self.get_multi_timeframe_data({'1m': self.volatility_window})['1m']
        volatility = self.calculate_volatility(hist_data) or 0.01
        size = self.calculate_position_size(balance, current_price, volatility)
        logger.debug(f"Volatility: {volatility}, Size: {size}")

        if signal > 0 and not self.position:
            self.place_order("BUY", size)
            self.position = 'long'
            self.entry_price = current_price
            self.position_size = size
            logger.info(f"Buy executed: {size} @ {current_price}")
        elif signal < 0 and not self.position:
            self.place_order("SELL", size)
            self.position = 'short'
            self.entry_price = current_price
            self.position_size = size
            logger.info(f"Sell executed: {size} @ {current_price}")
        elif self.position:
            self.manage_position(current_price)

    def run(self):
        if not self.transformer_model or not all(self.ensemble_models.values()):
            logger.warning("Models not initialized, training now")
            self._retrain_model_if_needed()
        while True:
            try:
                self._retrain_model_if_needed()
                self.execute_trade()
                logger.debug(f"Cycle completed, sleeping for {self.interval}s")
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Run loop error: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    try:
        client = BybitClient(api_key, api_secret, testnet=True)
        strategy = AdvancedTradingStrategy(client)
        strategy.run()
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
