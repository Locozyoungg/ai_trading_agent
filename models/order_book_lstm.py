# models/order_book_lstm.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Bidirectional, Attention, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from sklearn.preprocessing import RobustScaler
import os
import logging
from pathlib import Path
from collections import deque
import random
from typing import Dict, Optional

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMarketPredictor:
    def __init__(self, model_path: str, data_path: str, seq_length: int = 50, prediction_steps: int = 5, rl_enabled: bool = True):
        self.model_path = model_path if model_path.endswith('.h5') else model_path + '.h5'
        self.data_path = data_path
        self.seq_length = seq_length
        self.prediction_steps = prediction_steps
        self.ohlcv_features = 5
        self.order_book_features = 100
        self.scaler_ohlcv = RobustScaler()
        self.scaler_order_book = RobustScaler()
        self.is_fitted = False
        self.model = self._build_advanced_model()
        self.rl_enabled = rl_enabled
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

        logger.debug(f"Initializing AdvancedMarketPredictor with model_path={self.model_path}, data_path={self.data_path}")

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path)
                logger.info(f"Successfully loaded model weights from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {self.model_path}: {str(e)}", exc_info=True)
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)

    def _build_advanced_model(self) -> Model:
        logger.debug("Building advanced model with Transformer and LSTM")
        ohlcv_input = Input(shape=(self.seq_length, self.ohlcv_features), name='ohlcv_input')
        order_book_input = Input(shape=(self.order_book_features,), name='order_book_input')

        # Transformer-enhanced LSTM
        lstm_out = Bidirectional(LSTM(256, return_sequences=True))(ohlcv_input)
        lstm_out = LayerNormalization()(lstm_out)
        attn_out = MultiHeadAttention(num_heads=8, key_dim=64)(lstm_out, lstm_out)
        lstm_out = LSTM(128, return_sequences=True)(attn_out)
        attention_out = Attention()([lstm_out, lstm_out])
        lstm_out = GlobalAveragePooling1D()(attention_out)
        lstm_out = Dropout(0.3)(lstm_out)

        ob_dense = Dense(256, activation='relu')(order_book_input)
        ob_dense = LayerNormalization()(ob_dense)
        ob_dense = Dense(128, activation='relu')(ob_dense)
        ob_dense = Dropout(0.3)(ob_dense)

        combined = Concatenate()([lstm_out, ob_dense])
        dense = Dense(512, activation='relu')(combined)
        dense = Dropout(0.3)(dense)
        dense = Dense(256, activation='relu')(dense)
        output = Dense(self.prediction_steps, activation='linear')(dense)

        model = Model(inputs=[ohlcv_input, order_book_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005), loss='huber', metrics=['mae'])
        logger.info("Advanced model compiled successfully")
        return model

    def _preprocess_order_book(self, order_book: Dict) -> np.ndarray:
        logger.debug(f"Preprocessing order book data: bids={len(order_book.get('b', []))}")
        bids = np.array(order_book.get('b', [])[:25], dtype=float) if order_book.get('b') else np.zeros((25, 2))
        asks = np.array(order_book.get('a', [])[:25], dtype=float) if order_book.get('a') else np.zeros((25, 2))
        ob_data = np.concatenate([bids, asks]).flatten()
        if len(ob_data) < self.order_book_features:
            ob_data = np.pad(ob_data, (0, self.order_book_features - len(ob_data)), mode='constant')
        logger.debug(f"Processed order book data shape: {ob_data.shape}")
        return ob_data[:self.order_book_features]

    def update_data(self, ohlcv_data: pd.DataFrame, order_book_data: Dict):
        logger.debug(f"Updating data with OHLCV shape={ohlcv_data.shape}")
        try:
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in ohlcv_data.columns for col in required_cols):
                raise ValueError(f"OHLCV data missing required columns: {required_cols}")
            ob_processed = self._preprocess_order_book(order_book_data)
            ob_df = pd.DataFrame(ob_processed.reshape(1, -1), columns=[f"ob_{i}" for i in range(self.order_book_features)])
            combined = pd.concat([ohlcv_data.tail(1)[required_cols], ob_df], axis=1)
            combined.to_csv(self.data_path, mode='a', header=not os.path.exists(self.data_path), index=False)
            logger.info(f"Successfully updated training data at {self.data_path}")
        except Exception as e:
            logger.error(f"Data update failed: {str(e)}", exc_info=True)

    def train(self, ohlcv_data: np.ndarray, order_book_data: Dict, epochs: int = 10, batch_size: int = 32):
        logger.debug(f"Training model with OHLCV shape={ohlcv_data.shape}")
        try:
            if ohlcv_data.shape[1] != self.ohlcv_features + 1:
                raise ValueError(f"OHLCV data shape {ohlcv_data.shape} does not match expected")
            ohlcv_prices = ohlcv_data[-self.seq_length:, 1:6]
            if ohlcv_prices.shape[0] < self.seq_length:
                pad_length = self.seq_length - ohlcv_prices.shape[0]
                ohlcv_prices = np.pad(ohlcv_prices, ((pad_length, 0), (0, 0)), mode='edge')
            processed_ob = self._preprocess_order_book(order_book_data)
            if not self.is_fitted:
                self.scaler_ohlcv.fit(ohlcv_prices)
                self.scaler_order_book.fit(processed_ob.reshape(1, -1))
                self.is_fitted = True
                logger.info("Scalers fitted to OHLCV and order book data")
            X_ohlcv = self.scaler_ohlcv.transform(ohlcv_prices)[np.newaxis, :, :]
            X_order_book = self.scaler_order_book.transform(processed_ob.reshape(1, -1))
            y = self._prepare_targets(ohlcv_prices)
            y = y[np.newaxis, :]
            self.model.fit(
                {'ohlcv_input': X_ohlcv, 'order_book_input': X_order_book},
                y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            self.model.save_weights(self.model_path)
            logger.info(f"Model weights saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return False

    def _prepare_targets(self, ohlcv_data: np.ndarray) -> np.ndarray:
        logger.debug(f"Preparing targets from OHLCV data with shape={ohlcv_data.shape}")
        closes = ohlcv_data[:, 3]
        if len(closes) < self.prediction_steps:
            padded = np.pad(closes, (0, self.prediction_steps - len(closes)), mode='edge')
            return padded
        return closes[-self.prediction_steps:]

    def predict(self, ohlcv_data: np.ndarray, order_book_data: Dict) -> Optional[np.ndarray]:
        logger.debug(f"Predicting with OHLCV shape={ohlcv_data.shape}")
        try:
            if not self.is_fitted:
                logger.warning("Model not trained yet")
                return None
            if ohlcv_data.shape[1] != self.ohlcv_features + 1:
                raise ValueError(f"OHLCV data shape {ohlcv_data.shape} does not match expected")
            ohlcv_prices = ohlcv_data[-self.seq_length:, 1:6]
            if ohlcv_prices.shape[0] < self.seq_length:
                pad_length = self.seq_length - ohlcv_prices.shape[0]
                ohlcv_prices = np.pad(ohlcv_prices, ((pad_length, 0), (0, 0)), mode='edge')
            processed_ob = self._preprocess_order_book(order_book_data)
            X_ohlcv = self.scaler_ohlcv.transform(ohlcv_prices)[np.newaxis, :, :]
            X_order_book = self.scaler_order_book.transform(processed_ob.reshape(1, -1))
            prediction = self.model.predict({'ohlcv_input': X_ohlcv, 'order_book_input': X_order_book}, verbose=0)
            inverse_prediction = self.scaler_ohlcv.inverse_transform(prediction.reshape(1, -1))[0]
            logger.info(f"Prediction successful: {inverse_prediction}")
            return inverse_prediction
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return None

    def rl_learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        logger.debug(f"RL learning with state shape={state[0].shape}")
        if not self.rl_enabled:
            logger.debug("RL learning disabled")
            return
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) < 64:
            logger.debug(f"Memory buffer not full: {len(self.memory)}/64")
            return
        batch = random.sample(self.memory, 64)
        states_ohlcv, states_ob, actions, rewards, next_states_ohlcv, next_states_ob = [], [], [], [], [], []
        for s, a, r, ns in batch:
            states_ohlcv.append(s[0])
            states_ob.append(s[1])
            actions.append(a)
            rewards.append(r)
            next_states_ohlcv.append(ns[0])
            next_states_ob.append(ns[1])
        states_ohlcv = np.array(states_ohlcv)
        states_ob = np.array(states_ob)
        next_states_ohlcv = np.array(next_states_ohlcv)
        next_states_ob = np.array(next_states_ob)
        targets = self.model.predict({'ohlcv_input': states_ohlcv, 'order_book_input': states_ob}, verbose=0)
        next_q = self.model.predict({'ohlcv_input': next_states_ohlcv, 'order_book_input': next_states_ob}, verbose=0)
        for i, a in enumerate(actions):
            targets[i, a] = rewards[i] + self.gamma * np.max(next_q[i]) * (1 - int(rewards[i] == 0))
        self.model.fit(
            {'ohlcv_input': states_ohlcv, 'order_book_input': states_ob},
            targets,
            epochs=1,
            verbose=0
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.info(f"RL learning completed. Updated epsilon: {self.epsilon}")

    def rl_action(self, ohlcv_data: np.ndarray, order_book_data: Dict) -> int:
        logger.debug(f"Selecting RL action with OHLCV shape={ohlcv_data.shape}")
        if not self.rl_enabled or random.random() < self.epsilon:
            action = random.randint(0, self.prediction_steps - 1)
            logger.debug(f"Random action selected: {action}")
            return action
        preds = self.predict(ohlcv_data, order_book_data)
        action = np.argmax(preds) if preds is not None else 0
        logger.info(f"Predicted action: {action}")
        return action

if __name__ == "__main__":
    predictor = AdvancedMarketPredictor(
        model_path="C:/Users/MOTO/OneDrive/ai_trading_agent/saved_models/order_book_predictor",
        data_path="C:/Users/MOTO/OneDrive/ai_trading_agent/data/combined_market_BTCUSDT.csv"
    )
