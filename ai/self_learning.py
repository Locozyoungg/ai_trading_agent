# ai/self_learning.py
"""
State-of-the-Art Self-Learning Trading Agent
Combines deep LSTM networks with Double DQN for adaptive, high-performance trading.
"""

import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import logging
from typing import Optional, Tuple, Dict, List
from bybit_client import BybitClient
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AdvancedSelfLearning:
    def __init__(
        self,
        api: BybitClient,
        model_path: str = str(Path.home() / "OneDrive" / "ai_trading_agent" / "saved_models" / "advanced_trading_model.keras"),
        sequence_length: int = 50,
        ohlcv_features: int = 5,
        order_book_levels: int = 25,
        prediction_steps: int = 5
    ):
        self.api = api
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ohlcv_features = ohlcv_features
        self.order_book_levels = order_book_levels
        self.order_book_features = 2 * (2 * order_book_levels)
        self.prediction_steps = prediction_steps

        self.scaler_ohlcv = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_order_book = MinMaxScaler(feature_range=(-1, 1))
        self.is_scaler_fitted = {'ohlcv': False, 'order_book': False}

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 0.5  # Adjusted for faster exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

        self.current_position = None
        self.profit_target = 0.03
        self.stop_loss = 0.015
        self.trailing_stop = 0.01
        self.max_risk_percent = 1.5
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.trade_history = deque(maxlen=1000)

        self.model = self._load_or_build_model()
        self.target_model = self._load_or_build_model()
        self.target_model.set_weights(self.model.get_weights())
        self._pretrain_model()

    def _load_or_build_model(self) -> Model:
        try:
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                self._validate_model_shape(model)
                return model
            return self._build_advanced_model()
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}. Building new model.")
            return self._build_advanced_model()

    def _validate_model_shape(self, model: Model):
        expected_ohlcv_shape = (self.sequence_length, self.ohlcv_features)
        expected_ob_shape = (self.order_book_features,)
        model_inputs = model.input_shape
        if (len(model_inputs) < 2 or
            model_inputs[0][1:] != expected_ohlcv_shape or
            model_inputs[1][1:] != expected_ob_shape):
            logger.warning("Model shape mismatch. Rebuilding.")
            raise ValueError("Invalid model shape")

    def _build_advanced_model(self) -> Model:
        ohlcv_input = Input(shape=(self.sequence_length, self.ohlcv_features), name='ohlcv_input')
        ohlcv_lstm = Bidirectional(LSTM(256, return_sequences=True))(ohlcv_input)
        ohlcv_lstm = Attention()([ohlcv_lstm, ohlcv_lstm])
        ohlcv_lstm = LSTM(128)(ohlcv_lstm)
        ohlcv_lstm = Dropout(0.3)(ohlcv_lstm)

        order_book_input = Input(shape=(self.order_book_features,), name='order_book_input')
        ob_dense = Dense(256, activation='relu')(order_book_input)
        ob_dense = Dense(128, activation='relu')(ob_dense)
        ob_dense = Dropout(0.3)(ob_dense)

        combined = Concatenate()([ohlcv_lstm, ob_dense])
        dense = Dense(512, activation='relu')(combined)
        dense = Dropout(0.3)(dense)
        output = Dense(3, activation='linear')(dense)  # Buy, Sell, Hold Q-values

        model = Model(inputs=[ohlcv_input, order_book_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber')
        return model

    def _pretrain_model(self):
        if not os.path.exists(self.model_path):
            ohlcv_data = self.api.get_historical_data("BTCUSDT", limit=500)
            order_book = self.api.get_order_book("BTCUSDT")
            states_ohlcv = self._preprocess_ohlcv(ohlcv_data[-500:])
            states_ob = self._preprocess_order_book(order_book)
            dummy_targets = np.zeros((1, 3))
            self.model.fit(
                {'ohlcv_input': states_ohlcv[np.newaxis, :], 'order_book_input': states_ob[np.newaxis, :]},
                dummy_targets,
                epochs=50,
                verbose=0
            )
            self.model.save(self.model_path)
            logger.info("Model pretrained with 500 data points")

    def _preprocess_ohlcv(self, data: np.ndarray) -> np.ndarray:
        try:
            if data.shape[1] != self.ohlcv_features + 1:
                raise ValueError(f"OHLCV data expected {self.ohlcv_features + 1} columns, got {data.shape[1]}")
            prices = data[-self.sequence_length:, 1:6]
            if not self.is_scaler_fitted['ohlcv']:
                normalized = self.scaler_ohlcv.fit_transform(prices)
                self.is_scaler_fitted['ohlcv'] = True
            else:
                normalized = self.scaler_ohlcv.transform(prices)
            return normalized
        except Exception as e:
            logger.error(f"OHLCV preprocessing failed: {str(e)}")
            return np.zeros((self.sequence_length, self.ohlcv_features))

    def _preprocess_order_book(self, data: Dict) -> np.ndarray:
        try:
            bids = np.array(data.get('b', [])[:self.order_book_levels]).flatten()
            asks = np.array(data.get('a', [])[:self.order_book_levels]).flatten()
            ob_data = np.concatenate([bids, asks])
            if len(ob_data) < self.order_book_features:
                ob_data = np.pad(ob_data, (0, self.order_book_features - len(ob_data)), mode='constant')
            if not self.is_scaler_fitted['order_book']:
                normalized = self.scaler_order_book.fit_transform(ob_data.reshape(1, -1))
                self.is_scaler_fitted['order_book'] = True
            else:
                normalized = self.scaler_order_book.transform(ob_data.reshape(1, -1))
            return normalized[0]
        except Exception as e:
            logger.error(f"Order book preprocessing failed: {str(e)}")
            return np.zeros(self.order_book_features)

    def store_experience(self, state: Tuple[np.ndarray, np.ndarray], action: int, reward: float, next_state: Tuple[np.ndarray, np.ndarray], done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        try:
            batch = random.sample(self.memory, self.batch_size)
            states_ohlcv, states_ob, actions, rewards, next_states_ohlcv, next_states_ob, dones = [], [], [], [], [], [], []
            for s, a, r, ns, d in batch:
                states_ohlcv.append(s[0])
                states_ob.append(s[1])
                actions.append(a)
                rewards.append(r)
                next_states_ohlcv.append(ns[0])
                next_states_ob.append(ns[1])
                dones.append(d)

            states_ohlcv = np.array(states_ohlcv)
            states_ob = np.array(states_ob)
            next_states_ohlcv = np.array(next_states_ohlcv)
            next_states_ob = np.array(next_states_ob)

            targets = self.model.predict([states_ohlcv, states_ob], verbose=0)
            next_q_values = self.target_model.predict([next_states_ohlcv, next_states_ob], verbose=0)
            for i, a in enumerate(actions):
                if dones[i]:
                    targets[i, a] = rewards[i]
                else:
                    targets[i, a] = rewards[i] + self.gamma * np.max(next_q_values[i])

            self.model.fit(
                {'ohlcv_input': states_ohlcv, 'order_book_input': states_ob},
                targets,
                batch_size=self.batch_size,
                epochs=1,
                verbose=0
            )
            self.steps += 1
            if self.steps % self.target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())
                logger.info("Target network updated")
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        except Exception as e:
            logger.error(f"Replay failed: {str(e)}")

    def predict_action(self, analysis: Dict, position: Optional[Dict], balance: float) -> str:
        try:
            ohlcv_state = self.api.get_historical_data("BTCUSDT", limit=self.sequence_length)
            order_book_state = self.api.get_latest_order_book()
            current_price = self.api.get_current_price("BTCUSDT")
            volatility = analysis.get('volatility', 0.01)

            ohlcv_processed = self._preprocess_ohlcv(ohlcv_state)
            ob_processed = self._preprocess_order_book(order_book_state)
            state = (ohlcv_processed, ob_processed)

            if random.random() < self.epsilon:
                action = random.randint(0, 2)
            else:
                q_values = self.model.predict([ohlcv_processed[np.newaxis, :], ob_processed[np.newaxis, :]], verbose=0)[0]
                threshold = volatility * current_price * 0.005
                if q_values[0] - q_values[2] > threshold:
                    action = 0
                elif q_values[1] - q_values[2] > threshold:
                    action = 1
                else:
                    action = 2
                logger.debug(f"Q-values: {q_values}, Threshold: {threshold}, Action: {action}")

            actions = ["BUY", "SELL", "HOLD"]
            action_str = actions[action]

            if position and position.get('size', 0) > 0:
                self.current_position = (position['side'].capitalize(), position['entry_price'], position['size'], time.time())
                position_action = self.manage_position(current_price)
                if position_action == "CLOSE":
                    return "BUY" if self.current_position[0] == "Sell" else "SELL"
                return "HOLD"

            risk_amount = balance * (self.max_risk_percent / 100)
            size = min(risk_amount / (current_price * volatility), 0.1)
            size = max(size, 0.001)
            if size > 0.1:
                return "HOLD"

            return action_str
        except Exception as e:
            logger.error(f"Action prediction failed: {str(e)}")
            return "HOLD"

    def update_trade_state(self, side: str, entry_price: float, qty: float):
        if side == "Closed":
            self.current_position = None
        else:
            self.current_position = (side, entry_price, qty, time.time())
            self.total_trades += 1
            self.trade_history.append({"side": side, "entry_price": entry_price, "qty": qty, "timestamp": time.time()})
            logger.info(f"Trade opened: {side} {qty} @ {entry_price}")

    def clear_trade_state(self, exit_price: float) -> float:
        if not self.current_position:
            return 0.0
        try:
            side, entry_price, qty, _ = self.current_position
            profit = (exit_price - entry_price) * qty if side == "Buy" else (entry_price - exit_price) * qty
            reward = profit / entry_price
            if profit > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            self.trade_history.append({"exit_price": exit_price, "profit": profit, "timestamp": time.time()})
            logger.info(f"Trade closed: Profit {profit}, Total Trades: {self.total_trades}")
            self.current_position = None
            return reward
        except Exception as e:
            logger.error(f"Clear trade state failed: {str(e)}")
            return 0.0

    def manage_position(self, current_price: float) -> str:
        if not self.current_position:
            return "HOLD"
        side, entry_price, _, _ = self.current_position
        profit = (current_price - entry_price) / entry_price if side == "Buy" else (entry_price - current_price) / entry_price
        dynamic_profit_target = max(self.profit_target * self.api.get_historical_data("BTCUSDT", limit=50)[-1, 3] * 0.01, 0.02)
        dynamic_stop_loss = min(self.stop_loss * self.api.get_historical_data("BTCUSDT", limit=50)[-1, 3] * 0.01, 0.01)
        if profit >= dynamic_profit_target or profit <= -dynamic_stop_loss:
            return "CLOSE"
        return "HOLD"

    def train_episode(self, analysis: Dict, position: Optional[Dict], balance: float):
        try:
            ohlcv_data = self.api.get_historical_data("BTCUSDT", limit=self.sequence_length)
            order_book_data = self.api.get_latest_order_book()
            current_price = self.api.get_current_price("BTCUSDT")
            volatility = analysis.get('volatility', 0.01)

            state_ohlcv = self._preprocess_ohlcv(ohlcv_data)
            state_ob = self._preprocess_order_book(order_book_data)
            state = (state_ohlcv, state_ob)

            action = self.predict_action(analysis, position, balance)
            action_idx = ["BUY", "SELL", "HOLD"].index(action)

            next_ohlcv = self._preprocess_ohlcv(ohlcv_data[1:] if len(ohlcv_data) > self.sequence_length else ohlcv_data)
            next_ob = self._preprocess_order_book(order_book_data)
            next_state = (next_ohlcv, next_ob)

            reward = 0
            done = False
            if position and position.get('size', 0) > 0:
                self.current_position = (position['side'].capitalize(), position['entry_price'], position['size'], time.time())
                position_action = self.manage_position(current_price)
                if position_action == "CLOSE":
                    reward = self.clear_trade_state(current_price)
                    done = True
            elif action in ["BUY", "SELL"]:
                size = min(balance * (self.max_risk_percent / 100) / (current_price * volatility), 0.1)
                if size >= 0.001:
                    self.update_trade_state(action.capitalize(), current_price, size)

            self.store_experience(state, action_idx, reward, next_state, done)
            self.replay()
            self.save_model()
            logger.info("Self-learning episode completed successfully")
        except Exception as e:
            logger.error(f"Training episode failed: {str(e)}")

    def save_model(self):
        try:
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Save model failed: {str(e)}")

    def get_performance_metrics(self) -> Dict:
        win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
        return {
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": win_rate,
            "epsilon": self.epsilon
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api = BybitClient(os.getenv('BYBIT_API_KEY'), os.getenv('BYBIT_API_SECRET'), testnet=True)
    agent = AdvancedSelfLearning(api)
    analysis = {'volatility': 0.01}
    action = agent.predict_action(analysis, None, 1000.0)
    print(f"Predicted action: {action}")
