# main.py
"""
State-of-the-Art AI Trading System for Bybit
Integrates advanced prediction, risk management, multi-strategy execution, and report generation.
"""

import logging
import os
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import sys
from threading import Thread, Lock
from collections import deque
from datetime import datetime

load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

try:
    from bybit_client import BybitClient
    from ai.self_learning import AdvancedSelfLearning
    # Placeholder imports for other components (to be implemented separately)
    from risk_management.leverage_control import LeverageControl
    from risk_management.max_drawdown import MaxDrawdown
    from risk_management.max_loss import MaxLossPerTrade
    from risk_management.position_sizing import RiskManagement as PositionSizing
    from risk_management.risk_manager import RiskManager
    from risk_management.stop_loss_take_profit import StopLossTakeProfit
    from risk_management.trailing_stop import TrailingStopLoss
    from strategies.trading_strategy import AdvancedTradingStrategy
    from analysis.iceberg_detector import IcebergDetector
    from analysis.market_insights.market_insights import MarketInsights
    from analysis.ofi_analysis import OFIAnalysis
    from analysis.order_book_analysis import OrderBookAnalysis
    from analysis.order_timing import OrderTimingOptimizer
    from analysis.stop_hunt_detector import StopHuntDetector
    from models.order_book_lstm import AdvancedMarketPredictor
    from execution.hft_trading import HFTTrading
    from execution.market_maker import MarketMaker
    from execution.scalping_strategy import ScalpingStrategy
    from tracking.profit_tracker import ProfitTracker
    from tracking.strategy_report import StrategyReport
except ImportError as e:
    logger.critical(f"Failed to import required module: {e}", exc_info=True)
    sys.exit(1)

SAVED_MODELS_DIR = Path.home() / "OneDrive" / "ai_trading_agent" / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
SYMBOL_BTC = os.getenv('SYMBOL_BTC', 'BTCUSDT')
TRADE_SIZE_BTC = float(os.getenv('TRADE_SIZE_BTC', '0.01'))
MAX_POSITION_BTC = float(os.getenv('MAX_POSITION_BTC', '0.1'))
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '1000.0'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))

if not API_KEY or not API_SECRET:
    raise ValueError("API credentials missing!")

class TradingSystem:
    def __init__(self):
        logger.debug("Initializing TradingSystem")
        self.running = False
        self.active_strategy = 'default'
        self.client = BybitClient(API_KEY, API_SECRET, testnet=TESTNET)
        self.symbol = SYMBOL_BTC
        self.trade_size = TRADE_SIZE_BTC
        self.max_position = MAX_POSITION_BTC
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.order_book_model = AdvancedMarketPredictor(
            model_path=str(SAVED_MODELS_DIR / "order_book_predictor"),
            data_path=str(DATA_DIR / f"combined_market_{self.symbol}.csv"),
            seq_length=50,
            prediction_steps=5
        )
        self.position_info = {'size': 0.0, 'side': None, 'entry_price': 0.0, 'unrealised_pnl': 0.0, 'timestamp': None}
        self.trade_history = deque(maxlen=1000)
        self.volatility_window = deque(maxlen=50)
        self.lock = Lock()
        self.risk_components = {}
        self.analysis_components = {}
        self.trading_strategy = None
        self.execution_strategies = {}
        self.tracking_components = {}
        self.learning_components = {}
        self.last_report_time = time.time()
        self.report_interval = 3600
        self.last_position_sync_time = 0
        self.position_sync_interval = 60
        self.monitor_thread = None
        try:
            self.client.start_websocket()
            self._sync_open_positions()
            self._initialize_components()
            self.running = True
            self.monitor_thread = Thread(target=self._monitor_market, daemon=True)
            self.monitor_thread.start()
            logger.info("TradingSystem initialized with multi-threaded AI capabilities")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            self.running = False

    def _initialize_components(self):
        logger.debug("Initializing components")
        self.initial_balance = self.client.get_balance() or self.initial_balance
        self.current_balance = self.initial_balance
        self.risk_components = self._initialize_risk_management()
        self.analysis_components = self._initialize_analysis_tools()
        self.trading_strategy = self._initialize_trading_strategy()
        self.tracking_components = self._initialize_tracking_components()
        self.learning_components = self._initialize_learning_components()
        self._initialize_execution_strategies()
        self._initialize_order_book_model()
        logger.info("All components initialized successfully")

    def _initialize_risk_management(self) -> Dict:
        logger.debug("Initializing risk management components")
        components = {
            'leverage': LeverageControl(self.client),
            'drawdown': MaxDrawdown(self.client, self.initial_balance, max_drawdown=0.15),
            'max_loss': MaxLossPerTrade(self.client, self.initial_balance),
            'position_sizing': PositionSizing(self.client, self.initial_balance),
            'risk_manager': RiskManager(self.client, symbol=self.symbol, max_loss=0.02, volatility_threshold=0.5),
            'stop_loss': StopLossTakeProfit(self.client),
            'trailing_stop': TrailingStopLoss(self.client)
        }
        return components

    def _initialize_analysis_tools(self) -> Dict:
        logger.debug("Initializing analysis tools")
        components = {
            'iceberg_detector': IcebergDetector(self.client, self.symbol),
            'market_insights': MarketInsights(self.client, [self.symbol]),
            'ofi_analyzer': OFIAnalysis(self.client, self.symbol),
            'order_book_analyzer': OrderBookAnalysis(self.client, self.symbol),
            'order_timing': OrderTimingOptimizer(self.client, self.symbol),
            'stop_hunt_detector': StopHuntDetector(self.client, self.symbol),
            'market_insights_1h': MarketInsights(self.client, [self.symbol], timeframe='1h'),
        }
        return components

    def _initialize_trading_strategy(self) -> AdvancedTradingStrategy:
        logger.debug("Initializing trading strategy")
        strategy = AdvancedTradingStrategy(
            client=self.client,
            symbol=self.symbol,
            N=10,
            initial_threshold=0.2,
            interval=10,
            lookback_period=20,
            volatility_window=10,
            risk_per_trade=RISK_PER_TRADE,
            stop_loss_factor=0.015,
            take_profit_factor=0.03,
            lstm_sequence_length=60
        )
        return strategy

    def _initialize_tracking_components(self) -> Dict:
        logger.debug("Initializing tracking components")
        components = {
            'profit_tracker': ProfitTracker(self.client, self.symbol),
            'strategy_report': StrategyReport(self.client)
        }
        return components

    def _initialize_learning_components(self) -> Dict:
        logger.debug("Initializing learning components")
        try:
            components = {
                'self_learning': AdvancedSelfLearning(
                    api=self.client,
                    model_path=str(SAVED_MODELS_DIR / "advanced_trading_model.keras"),
                    sequence_length=50
                )
            }
            return components
        except Exception as e:
            logger.error(f"Self-learning initialization failed: {str(e)}", exc_info=True)
            return {'self_learning': None}

    def _initialize_execution_strategies(self):
        logger.debug("Initializing execution strategies")
        common_args = {'position_info': self.position_info, 'risk_components': self.risk_components}
        self.execution_strategies = {
            'hft': HFTTrading(self.client, self.symbol, **common_args),
            'market_making': MarketMaker(self.client, self.symbol, **common_args),
            'scalping': ScalpingStrategy(self.client, self.symbol, **common_args),
            'default': self.trading_strategy
        }

    def _initialize_order_book_model(self):
        logger.debug("Initializing order book model")
        if not os.path.exists(self.order_book_model.model_path):
            ohlcv_data = self.client.get_historical_data(self.symbol, limit=200)
            order_book = self.client.get_order_book(self.symbol)
            self.order_book_model.train(ohlcv_data, order_book, epochs=50)
            logger.info("Order book model trained and initialized")
        else:
            logger.info("Order book model weights already exist. Skipping initial training")

    def _sync_open_positions(self):
        logger.debug("Syncing open positions")
        try:
            positions = self.client.get_positions(self.symbol)
            if not positions:
                logger.warning("No position data returned from API")
                return
            position = next((p for p in positions if float(p.get('contracts', 0)) > 0), None)
            with self.lock:
                if position:
                    self.position_info.update({
                        'size': float(position['contracts']),
                        'side': position['side'].lower(),
                        'entry_price': float(position['entryPrice']),
                        'unrealised_pnl': float(position['unrealisedPnl']),
                        'timestamp': datetime.fromtimestamp(int(position['timestamp']) / 1000)
                    })
                    logger.info(f"Synced open position: {self.position_info}")
                else:
                    self.position_info.update({'size': 0, 'side': None, 'entry_price': 0, 'unrealised_pnl': 0, 'timestamp': None})
                    logger.debug("No open positions found")
                self.current_balance = self.client.get_balance() or self.current_balance
                self.last_position_sync_time = time.time()
        except Exception as e:
            logger.error(f"Position sync failed: {str(e)}", exc_info=True)

    def _fetch_ohlcv_data(self, limit: int = 50) -> np.ndarray:
        logger.debug(f"Fetching OHLCV data: limit={limit}")
        try:
            data = self.client.get_historical_data(self.symbol, interval="1", limit=limit)
            if data.shape[0] < limit:
                logger.warning(f"Insufficient OHLCV data: {data.shape[0]} < {limit}")
                padded = np.zeros((limit, 6))
                padded[-data.shape[0]:] = data
                return padded
            self.volatility_window.append(np.std(data[:, 3]))
            return data
        except Exception as e:
            logger.error(f"OHLCV fetch failed: {str(e)}", exc_info=True)
            return np.zeros((limit, 6))

    def _monitor_market(self):
        logger.debug("Starting market monitoring thread")
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_position_sync_time >= self.position_sync_interval:
                    self._sync_open_positions()
                self._update_and_train_model()
                self._generate_reports()
                time.sleep(30)
            except Exception as e:
                logger.error(f"Market monitor failed: {str(e)}", exc_info=True)
                time.sleep(5)

    def _update_and_train_model(self):
        logger.debug("Updating and training order book model")
        try:
            ohlcv_data = self._fetch_ohlcv_data(limit=50)
            order_book = self.client.get_latest_order_book()
            ohlcv_df = pd.DataFrame(ohlcv_data[:, 1:6], columns=["open", "high", "low", "close", "volume"])
            self.order_book_model.update_data(ohlcv_df, order_book)
            self.order_book_model.train(ohlcv_data, order_book, epochs=10)
            logger.info("Model updated and retrained successfully")
        except Exception as e:
            logger.error(f"Model update/train failed: {str(e)}", exc_info=True)

    def _calculate_dynamic_size(self, current_price: float) -> float:
        logger.debug(f"Calculating dynamic position size: current_price={current_price}")
        with self.lock:
            volatility = np.mean(self.volatility_window) if self.volatility_window else 0.01
            risk_amount = self.current_balance * RISK_PER_TRADE
            available_size = self.max_position - self.position_info['size']
            size = min(risk_amount / (current_price * volatility), available_size)
            dynamic_size = max(min(size, self.trade_size), 0.001)
            logger.info(f"Dynamic size calculated: {dynamic_size} (volatility={volatility}, risk_amount={risk_amount})")
            return dynamic_size

    def _analyze_market_conditions(self) -> Dict:
        logger.debug("Analyzing market conditions")
        try:
            trades = self.client.get_latest_trades()
            ohlcv_data = self._fetch_ohlcv_data(limit=10)
            ohlcv_df = pd.DataFrame(ohlcv_data[:, 1:6], columns=["open", "high", "low", "close", "volume"])
            order_book = self.client.get_latest_order_book()
            predictions = self.order_book_model.predict(ohlcv_data, order_book)
            volatility = np.std(ohlcv_df['close']) / np.mean(ohlcv_df['close']) if len(ohlcv_df) > 1 else 0.01
            analysis = {
                'insights': self.analysis_components['market_insights'].run(trades=trades),
                'ofi': self.analysis_components['ofi_analyzer'].compute_order_flow_imbalance(trades=trades),
                'icebergs': self.analysis_components['iceberg_detector'].detect_iceberg_orders(),
                'ob_prediction': predictions,
                'stop_hunt': self.analysis_components['stop_hunt_detector'].detect_stop_hunts(ohlcv_df),
                'order_timing': self.analysis_components['order_timing'].detect_large_orders(),
                'volatility': volatility
            }
            logger.info(f"Market analysis completed: volatility={volatility}, predictions={predictions}")
            return analysis
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}", exc_info=True)
            return {'ob_prediction': None, 'volatility': 0.01, 'stop_hunt': False, 'ofi': 0.0}

    def _generate_reports(self):
        logger.debug("Generating reports")
        try:
            current_time = time.time()
            if current_time - self.last_report_time < self.report_interval:
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            perf_metrics = self.learning_components['self_learning'].get_performance_metrics()
            perf_report = (
                f"Performance Report - {timestamp}\n"
                f"Total Trades: {perf_metrics['total_trades']}\n"
                f"Wins: {perf_metrics['win_count']}\n"
                f"Losses: {perf_metrics['loss_count']}\n"
                f"Win Rate: {perf_metrics['win_rate']:.2%}\n"
                f"Epsilon: {perf_metrics['epsilon']:.4f}\n"
                f"Current Balance: {self.current_balance:.2f} USDT\n"
                f"Unrealised PnL: {self.position_info['unrealised_pnl']:.2f}\n"
            )
            with open(REPORTS_DIR / f"performance_{timestamp}.txt", "w", encoding='utf-8') as f:
                f.write(perf_report)

            analysis = self._analyze_market_conditions()
            preds = analysis.get('ob_prediction', [self.client.get_current_price(self.symbol) or 0])
            current_price = self.client.get_current_price(self.symbol) or 0
            trend = "Up" if preds[-1] > current_price else "Down"
            recs = (
                f"Recommendations - {timestamp}\n"
                f"Predicted Trend: {trend}\n"
                f"Volatility: {analysis['volatility']:.4f}\n"
                f"Order Flow Imbalance: {analysis['ofi']:.2f}\n"
                f"Stop Hunt Detected: {analysis['stop_hunt']}\n"
                f"Recommended Strategy: {self.active_strategy}\n"
            )
            with open(REPORTS_DIR / f"recommendations_{timestamp}.txt", "w", encoding='utf-8') as f:
                f.write(recs)

            action = self.learning_components['self_learning'].predict_action(analysis, self.position_info, self.current_balance)
            signals = (
                f"Trading Signals - {timestamp}\n"
                f"Action: {action}\n"
                f"Current Price: {current_price:.2f}\n"
                f"Predicted Prices (Next {self.order_book_model.prediction_steps} steps): {', '.join([f'{p:.2f}' for p in preds])}\n"
                f"Position: {self.position_info['side']} {self.position_info['size']} @ {self.position_info['entry_price']}\n"
            )
            with open(REPORTS_DIR / f"signals_{timestamp}.txt", "w", encoding='utf-8') as f:
                f.write(signals)

            self.last_report_time = current_time
            logger.info(f"Generated reports at {REPORTS_DIR} for {timestamp}")
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}", exc_info=True)

    def _make_trade_decision(self, analysis: Dict):
        logger.debug(f"Making trade decision with analysis: {analysis}")
        try:
            self._sync_open_positions()
            current_price = self.client.get_current_price(self.symbol)
            if not current_price:
                logger.error("Failed to fetch current price. Skipping trade decision.")
                return

            action = self.learning_components['self_learning'].predict_action(
                analysis, self.position_info, self.current_balance
            )
            logger.info(f"Predicted action: {action}")

            strategy = self.execution_strategies[self.active_strategy]
            with self.lock:
                if self.position_info['size'] > 0:
                    if action in ["SELL", "BUY"] and (
                        (action == "SELL" and self.position_info['side'] == 'long') or
                        (action == "BUY" and self.position_info['side'] == 'short')
                    ):
                        order = strategy.close_position() if hasattr(strategy, 'close_position') else self.client.close_position(self.symbol)
                        if order and 'id' in order:
                            profit = self.position_info['unrealised_pnl']
                            self.trade_history.append({
                                "type": "close",
                                "side": self.position_info['side'],
                                "size": self.position_info['size'],
                                "profit": profit,
                                "timestamp": datetime.now()
                            })
                            reward = self.learning_components['self_learning'].clear_trade_state(current_price)
                            self.learning_components['self_learning'].train_episode(analysis, self.position_info, self.current_balance)
                            logger.info(f"Closed {self.position_info['side']} position: Profit {profit}, Reward {reward}")
                            self._sync_open_positions()
                        else:
                            logger.error(f"Failed to close position: {order}")
                            raise RuntimeError(f"Close position failed: {order}")
                elif action in ["BUY", "SELL"]:
                    size = self._calculate_dynamic_size(current_price)
                    if size > 0:
                        order = strategy.execute_trade(action.lower(), size, current_price) if hasattr(strategy, 'execute_trade') else self.client.place_order(self.symbol, size, action)
                        if order and 'id' in order:
                            self.trade_history.append({
                                "type": "open",
                                "side": action.lower(),
                                "size": size,
                                "price": current_price,
                                "timestamp": datetime.now()
                            })
                            self.learning_components['self_learning'].update_trade_state(action.capitalize(), current_price, size)
                            self.learning_components['self_learning'].train_episode(analysis, self.position_info, self.current_balance)
                            logger.info(f"Opened {action.lower()} position: {size} @ {current_price}")
                            self._sync_open_positions()
                        else:
                            logger.error(f"Failed to place {action} order: {order}")
                            raise RuntimeError(f"Trade execution failed: {order}")
        except Exception as e:
            logger.error(f"Trade decision failed: {str(e)}", exc_info=True)
            raise

    def _select_execution_strategy(self, analysis: Dict) -> str:
        logger.debug(f"Selecting execution strategy based on analysis: {analysis}")
        if analysis.get('stop_hunt'):
            return 'hft'
        elif abs(analysis.get('ofi', 0)) > 0.3:
            return 'market_making'
        elif analysis.get('volatility', 0) > 0.05:
            return 'scalping'
        return 'default'

    def _switch_strategy(self, new_strategy: str):
        logger.debug(f"Switching strategy to {new_strategy}")
        with self.lock:
            if self.active_strategy != new_strategy:
                logger.info(f"Strategy switched from {self.active_strategy} to {new_strategy}")
                self.active_strategy = new_strategy

    def run(self):
        logger.info("Starting state-of-the-art trading system...")
        if not self.running:
            logger.error("System not running due to initialization failure")
            return
        try:
            while self.running:
                analysis = self._analyze_market_conditions()
                self._switch_strategy(self._select_execution_strategy(analysis))
                self._make_trade_decision(analysis)
                logger.info("Trade cycle completed")
                time.sleep(15)
        except KeyboardInterrupt:
            logger.info("Shutting down due to keyboard interrupt...")
            self.shutdown()
        except Exception as e:
            logger.critical(f"Run error: {str(e)}", exc_info=True)
            self.shutdown()

    def shutdown(self):
        logger.debug("Initiating system shutdown")
        self.running = False
        try:
            self.client.stop_websocket()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitor thread did not terminate gracefully")
            self._generate_reports()
            logger.info("Trading system shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown failed: {str(e)}", exc_info=True)
        finally:
            sys.exit(0)

if __name__ == "__main__":
    trading_system = TradingSystem()
    if trading_system.running:
        trading_system.run()
    else:
        logger.error("Initialization failed. System not started")
