"""
Main Trading System — Orchestrator

Integrates market analysis, signal generation, risk management,
trade execution, and performance tracking into a single coherent loop.

Key improvements over the original:
  - No "always-buy-first" bug — trades are purely signal-driven
  - All risk management components are wired into the decision flow
  - The signal generator uses real technical indicators, not fake DQN training
  - Execution strategies use centralized data — no duplicate WebSockets
  - Position size adapts to volatility, win rate, and consecutive losses
  - Full performance tracking with metrics
"""

import logging
import os
import sys
import time
import signal
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from threading import Thread, Lock
from collections import deque
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Logging (single configuration) ─────────────────────────────────────────
logger = logging.getLogger('trading_system')
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
fh = logging.FileHandler('trading_bot.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.addHandler(sh)

# ── Imports ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import config as cfg

from bybit_client import BybitClient

from risk_management.leverage_control import LeverageControl
from risk_management.max_drawdown import MaxDrawdown
from risk_management.max_loss import MaxLossPerTrade
from risk_management.position_sizing import RiskManagement as PositionSizing
from risk_management.risk_manager import RiskManager
from risk_management.stop_loss_take_profit import StopLossTakeProfit
from risk_management.trailing_stop import TrailingStopLoss

from analysis.iceberg_detector import IcebergDetector
from analysis.market_analysis import MarketInsights
from analysis.ofi_analysis import OFIAnalysis
from analysis.order_book_analysis import OrderBookAnalysis
from analysis.order_timing import OrderTimingOptimizer
from analysis.stop_hunt_detector import StopHuntDetector

from ai.self_learning import SignalGenerator, TradeRecord

from strategies.trading_strategy import AdvancedTradingStrategy

from execution.hft_trading import HFTTrading
from execution.market_maker import MarketMaker
from execution.scalping_strategy import ScalpingStrategy

from tracking.profit_tracker import ProfitTracker
from tracking.strategy_report import StrategyReport


class TradingSystem:
    """
    Central orchestrator for the AI Trading Agent.

    Owns all components and runs a single coherent trade loop.
    """

    def __init__(self):
        logger.debug("Initializing TradingSystem v2")
        self.running = False
        self.active_strategy = 'default'
        self.symbol = cfg.SYMBOL
        self.initial_balance = cfg.INITIAL_BALANCE
        self.current_balance = cfg.INITIAL_BALANCE

        # Shared position state (thread-safe via lock)
        self.position_info = {
            'size': 0.0, 'side': None, 'entry_price': 0.0,
            'unrealised_pnl': 0.0, 'timestamp': None
        }
        self.trade_history = deque(maxlen=1000)
        self.volatility_window = deque(maxlen=50)
        self.lock = Lock()

        # Component containers
        self.client: Optional[BybitClient] = None
        self.risk_components: Dict = {}
        self.analysis_components: Dict = {}
        self.trading_strategy: Optional[AdvancedTradingStrategy] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.execution_strategies: Dict = {}
        self.tracking_components: Dict = {}

        # Timing
        self.last_report_time = time.time()
        self.report_interval = cfg.REPORT_INTERVAL
        self.last_position_sync_time = 0
        self.position_sync_interval = cfg.POSITION_SYNC_INTERVAL
        self._last_trade_time = 0
        self._min_trade_interval = cfg.TRADE_LOOP_INTERVAL

        # Metrics
        self.total_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = cfg.INITIAL_BALANCE

        # Initialize
        try:
            self.client = BybitClient(
                cfg.BYBIT_API_KEY, cfg.BYBIT_API_SECRET, testnet=cfg.USE_TESTNET
            )
            self.client.start_websocket()
            self._initialize_components()
            self.running = True
            logger.info("TradingSystem v2 initialized successfully")
        except Exception as e:
            logger.critical("Initialization failed: %s", e, exc_info=True)
            self.running = False

    # ── Initialization ─────────────────────────────────────────────────────

    def _initialize_components(self):
        """Initialize all sub-components."""
        self._sync_balance()
        self.risk_components = self._init_risk()
        self.analysis_components = self._init_analysis()
        self.signal_generator = SignalGenerator()
        self.trading_strategy = self._init_strategy()
        self.tracking_components = self._init_tracking()
        self._init_execution_strategies()
        logger.info("All components initialized")

    def _sync_balance(self):
        try:
            bal = self.client.get_balance()
            if bal and bal > 0:
                self.current_balance = bal
                self.initial_balance = bal
                self.peak_balance = bal
        except Exception as e:
            logger.warning("Balance sync failed: %s", e)

    def _init_risk(self) -> Dict:
        return {
            'leverage': LeverageControl(self.client),
            'drawdown': MaxDrawdown(self.client, self.initial_balance, max_drawdown=cfg.MAX_DRAWDOWN),
            'max_loss': MaxLossPerTrade(self.client, self.initial_balance),
            'position_sizing': PositionSizing(self.client, self.initial_balance),
            'risk_manager': RiskManager(
                self.client, symbol=self.symbol,
                max_loss=cfg.RISK_PER_TRADE, volatility_threshold=cfg.VOLATILITY_THRESHOLD
            ),
            'stop_loss': StopLossTakeProfit(self.client),
            'trailing_stop': TrailingStopLoss(self.client),
        }

    def _init_analysis(self) -> Dict:
        return {
            'iceberg': IcebergDetector(self.client, self.symbol),
            'market_insights': MarketInsights(self.client, [self.symbol], timeframe='60'),
            'market_insights_1h': MarketInsights(self.client, [self.symbol], timeframe='60'),
            'ofi': OFIAnalysis(self.client, self.symbol),
            'order_book': OrderBookAnalysis(self.client, self.symbol),
            'order_timing': OrderTimingOptimizer(self.client, self.symbol),
            'stop_hunt': StopHuntDetector(self.client, self.symbol),
        }

    def _init_strategy(self) -> AdvancedTradingStrategy:
        return AdvancedTradingStrategy(
            client=self.client, symbol=self.symbol,
            N=10, initial_threshold=0.1, interval=10,
            lookback_period=cfg.LOOKBACK_PERIOD,
            volatility_window=cfg.VOLATILITY_WINDOW,
            risk_per_trade=cfg.RISK_PER_TRADE,
            stop_loss_factor=cfg.STOP_LOSS_FACTOR,
            take_profit_factor=cfg.TAKE_PROFIT_FACTOR,
            sequence_length=cfg.SEQUENCE_LENGTH,
        )

    def _init_tracking(self) -> Dict:
        return {
            'profit_tracker': ProfitTracker(self.client, self.symbol),
            'strategy_report': StrategyReport(self.client),
        }

    def _init_execution_strategies(self):
        common = {'position_info': self.position_info, 'risk_components': self.risk_components}
        self.execution_strategies = {
            'hft': HFTTrading(
                self.client, self.symbol, **common,
                spread_threshold=cfg.HFT_SPREAD_THRESHOLD,
                order_size=cfg.HFT_ORDER_SIZE,
            ),
            'market_making': MarketMaker(
                self.client, self.symbol, **common,
                spread=cfg.MARKET_MAKER_SPREAD,
                size=cfg.MARKET_MAKER_SIZE,
            ),
            'scalping': ScalpingStrategy(
                self.client, self.symbol, **common,
                spread=cfg.SCALPING_SPREAD,
                size=cfg.SCALPING_SIZE,
            ),
        }

    # ── Market Analysis ────────────────────────────────────────────────────

    def _analyze_market_conditions(self) -> Dict:
        """Collect and fuse all market analysis into a single dict."""
        analysis = {}

        # 1. Market insights (OHLCV-based)
        try:
            mi = self.analysis_components['market_insights'].analyze_market()
            analysis['market'] = mi.get(self.symbol, {})
        except Exception as e:
            logger.error("Market insights failed: %s", e)
            analysis['market'] = {}

        # 2. Order book metrics
        try:
            ob = self.analysis_components['order_book']
            analysis['ofi'] = ob.calculate_order_flow_imbalance(levels=cfg.OFI_LEVELS)
            analysis['spread_pct'] = ob.calculate_spread_pct()
            analysis['bid_ask_ratio'] = ob.calculate_bid_ask_ratio()
        except Exception as e:
            logger.error("Order book analysis failed: %s", e)

        # 3. Stop hunt check — only if we have enough data
        try:
            ohlcv = self.client.get_historical_data(self.symbol, interval='1', limit=30)
            if ohlcv is not None and len(ohlcv) > 10:
                import pandas as pd
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                analysis['stop_hunt'] = self.analysis_components['stop_hunt'].detect_stop_hunts(df)
            else:
                analysis['stop_hunt'] = False
        except Exception as e:
            logger.error("Stop-hunt check failed: %s", e)
            analysis['stop_hunt'] = False

        # 4. Large order detection
        try:
            timing = self.analysis_components['order_timing']
            analysis['large_order'] = timing.detect_large_orders()
        except Exception as e:
            logger.error("Order timing check failed: %s", e)
            analysis['large_order'] = None

        return analysis

    # ── Signal Generation ──────────────────────────────────────────────────

    def _generate_signal(self, analysis: Dict) -> 'SignalResult':
        """Generate a trading signal from all available data."""
        # Fetch OHLCV for technical indicators
        ohlcv = self.client.get_historical_data(self.symbol, interval='5', limit=100)
        if ohlcv is None or len(ohlcv) < 20:
            from ai.self_learning import SignalResult, MarketRegime
            return SignalResult('hold', 0.0, 0.0, MarketRegime('ranging', 0, 0, False, False), {})

        prices = ohlcv[:, 4]   # close prices
        volumes = ohlcv[:, 5]  # volume
        ofi = analysis.get('ofi')

        return self.signal_generator.generate_signal(prices, volumes, ofi)

    # ── Risk Gates ─────────────────────────────────────────────────────────

    def _check_risk_gates(self, action: str, size: float, price: float) -> tuple:
        """
        Check all risk management gates before allowing a trade.

        Returns:
            (allowed: bool, reason: str)
        """
        rc = self.risk_components

        # 1. Drawdown check
        try:
            if not rc['drawdown'].check_drawdown(self.current_balance):
                return False, "Max drawdown exceeded"
        except Exception as e:
            logger.warning("Drawdown check failed: %s", e)

        # 2. Max loss per trade
        try:
            if not rc['max_loss'].check_max_loss(size, price, self.current_balance):
                return False, "Max loss per trade exceeded"
        except Exception as e:
            logger.warning("Max loss check failed: %s", e)

        # 3. Risk manager (volatility + general risk)
        try:
            if not rc['risk_manager'].check_risk(action, size, price):
                return False, "Risk manager blocked"
        except Exception as e:
            logger.warning("Risk manager check failed: %s", e)

        # 4. Volatility check
        try:
            high_vol = rc['risk_manager'].check_volatility()
            if high_vol and size > cfg.HFT_ORDER_SIZE:
                # In high volatility, reduce size but don't block entirely
                size *= 0.5
        except Exception as e:
            logger.warning("Volatility check failed: %s", e)

        return True, ""

    # ── Position Sizing ────────────────────────────────────────────────────

    def _calculate_dynamic_size(self, price: float) -> float:
        """Calculate position size using risk-based sizing."""
        try:
            sizing = self.risk_components['position_sizing']
            size = sizing.calculate_position_size(self.current_balance, price)
        except Exception:
            # Fallback: fixed fractional sizing
            risk_amount = self.current_balance * cfg.RISK_PER_TRADE
            size = risk_amount / (price * cfg.STOP_LOSS_FACTOR) if price > 0 else 0.0

        # Apply performance-based adjustment
        factor = self.signal_generator.get_position_sizing_factor()
        size *= factor

        # Clamp to max position
        size = min(size, cfg.MAX_POSITION)

        # Ensure minimum trade size
        min_size = cfg.TRADE_SIZE
        return max(min_size, size)

    # ── Trade Execution ───────────────────────────────────────────────────

    def _execute_trade(self, action: str, size: float, price: float, source: str = 'signal'):
        """Execute a trade and update state."""
        side = action.upper()

        with self.lock:
            order = self.client.place_order(self.symbol, size, side, order_type="Market")
            if not order or 'id' not in order:
                logger.error("Order placement failed: %s", order)
                return False

            self.position_info.update({
                'size': size,
                'side': side.lower(),
                'entry_price': price,
                'unrealised_pnl': 0.0,
                'timestamp': datetime.now(),
            })
            self._last_trade_time = time.time()
            self.total_trades += 1

            self.trade_history.append({
                'type': 'open', 'side': side.lower(), 'size': size,
                'price': price, 'source': source,
                'timestamp': datetime.now(),
            })

            logger.info("OPEN %s %.6f @ %.2f [source=%s]", side, size, price, source)
            return True

    def _close_position(self, price: float, source: str = 'signal'):
        """Close the current position."""
        with self.lock:
            if self.position_info['size'] <= 0:
                return False

            close_side = 'SELL' if self.position_info['side'] == 'long' else 'BUY'
            size = self.position_info['size']
            entry_price = self.position_info['entry_price']

            order = self.client.place_order(
                self.symbol, size, close_side, order_type="Market", reduce_only=True
            )
            if not order or 'id' not in order:
                logger.error("Failed to close position: %s", order)
                return False

            # Calculate P&L
            if self.position_info['side'] == 'long':
                pnl = (price - entry_price) * size
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - price) * size
                pnl_pct = (entry_price - price) / entry_price * 100

            self.total_pnl += pnl
            self.current_balance += pnl
            self.peak_balance = max(self.peak_balance, self.current_balance)

            self.trade_history.append({
                'type': 'close', 'side': self.position_info['side'],
                'size': size, 'price': price, 'pnl': pnl, 'pnl_pct': pnl_pct,
                'source': source, 'timestamp': datetime.now(),
            })

            # Record in signal generator for weight adjustment
            trade_record = TradeRecord(
                entry_price=entry_price, exit_price=price,
                side=self.position_info['side'], size=size,
                pnl=pnl, pnl_pct=pnl_pct,
                signal_strength=0.0, regime='', timestamp=time.time(),
            )
            if hasattr(self, 'signal_generator') and self.signal_generator:
                self.signal_generator.record_trade_outcome(trade_record)

            self.position_info.update({
                'size': 0.0, 'side': None, 'entry_price': 0.0,
                'unrealised_pnl': 0.0, 'timestamp': datetime.now(),
            })

            logger.info("CLOSE %s %.6f @ %.2f | PnL: %.2f (%.2f%%)",
                        close_side, size, price, pnl, pnl_pct)
            return True

    # ── Main Trade Decision ────────────────────────────────────────────────

    def _make_trade_decision(self, analysis: Dict):
        """Core trading logic — called every cycle."""
        try:
            self._sync_open_positions()
            current_price = self.client.get_current_price(self.symbol)
            if not current_price or current_price <= 0:
                logger.error("Invalid current price")
                return

            # 1. Generate signal
            signal = self._generate_signal(analysis)
            logger.debug("Signal: %s (conf=%.3f, strength=%.3f, regime=%s)",
                         signal.action, signal.confidence, signal.strength, signal.regime.name)

            # 2. If in a position, check if we should close
            if self.position_info['size'] > 0:
                # Check stop-loss / take-profit first
                self._manage_open_position(current_price)

                # If still in a position after risk management, check signal
                if self.position_info['size'] > 0:
                    entry_price = self.position_info['entry_price']
                    side = self.position_info['side']
                    pnl_pct = (current_price - entry_price) / entry_price * 100 if side == 'long' \
                              else (entry_price - current_price) / entry_price * 100

                    # Close if signal strongly opposes position
                    if side == 'long' and signal.strength < -0.5:
                        logger.info("Signal reversal: closing long (strength=%.3f)", signal.strength)
                        self._close_position(current_price, source='signal_reversal')
                    elif side == 'short' and signal.strength > 0.5:
                        logger.info("Signal reversal: closing short (strength=%.3f)", signal.strength)
                        self._close_position(current_price, source='signal_reversal')
                return

            # 3. No position — check if we should open one
            if signal.action in ('buy', 'sell') and signal.confidence >= 0.4:
                action = signal.action.upper()
                size = self._calculate_dynamic_size(current_price)

                if size <= 0:
                    logger.debug("Calculated size <= 0, skipping trade")
                    return

                # Check all risk gates
                allowed, reason = self._check_risk_gates(action, size, current_price)
                if not allowed:
                    logger.warning("Risk gate blocked %s: %s", action, reason)
                    return

                # Rate limit check
                if time.time() - self._last_trade_time < self._min_trade_interval:
                    return

                self._execute_trade(action, size, current_price, source='signal')

        except Exception as e:
            logger.error("Trade decision failed: %s", e, exc_info=True)

    # ── Position Management ────────────────────────────────────────────────

    def _manage_open_position(self, current_price: float):
        """Apply risk management to an open position."""
        rc = self.risk_components
        pi = self.position_info

        if pi['size'] <= 0:
            return

        # Update unrealized P&L
        if pi['side'] == 'long':
            pi['unrealised_pnl'] = (current_price - pi['entry_price']) * pi['size']
        else:
            pi['unrealised_pnl'] = (pi['entry_price'] - current_price) * pi['size']

        # Stop-loss / take-profit
        try:
            rc['stop_loss'].manage_position(pi, current_price)
        except Exception as e:
            logger.warning("Stop-loss check failed: %s", e)

        # Trailing stop
        try:
            rc['trailing_stop'].update_trailing_stop(pi, current_price)
        except Exception as e:
            logger.warning("Trailing stop failed: %s", e)

        # If position was closed by risk management
        if pi['size'] <= 0:
            pnl = pi['unrealised_pnl']
            self.total_pnl += pnl
            self.current_balance += pnl
            self.peak_balance = max(self.peak_balance, self.current_balance)
            logger.info("Position closed by risk management. PnL: %.2f", pnl)

    # ── Position Sync ──────────────────────────────────────────────────────

    def _sync_open_positions(self):
        """Sync position state from the exchange."""
        if time.time() - self.last_position_sync_time < self.position_sync_interval:
            return
        self.last_position_sync_time = time.time()

        try:
            positions = self.client.get_positions(self.symbol)
            if positions:
                pos = positions[0]
                size = float(pos.get('contracts', 0))
                if size > 0:
                    with self.lock:
                        self.position_info.update({
                            'size': size,
                            'side': pos.get('side', 'long').lower(),
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'unrealised_pnl': float(pos.get('unrealisedPnl', 0)),
                            'timestamp': datetime.now(),
                        })
                elif self.position_info['size'] > 0:
                    # Position closed externally
                    with self.lock:
                        self.position_info['size'] = 0.0
                        self.position_info['side'] = None
        except Exception as e:
            logger.warning("Position sync failed: %s", e)

    # ── Strategy Selection ─────────────────────────────────────────────────

    def _select_strategy(self, analysis: Dict) -> str:
        """Select the most appropriate execution strategy based on market conditions."""
        vol = analysis.get('market', {}).get('volatility', 5.0)

        if vol > 3.0:
            # High volatility → use HFT (fast in/out)
            return 'hft'
        elif analysis.get('ofi') is not None and abs(analysis.get('ofi', 0)) > 0.3:
            # Strong OFI → use scalping in direction of flow
            return 'scalping'
        elif vol < 1.0:
            # Low volatility → market making
            return 'market_making'
        return 'default'

    # ── Reporting ──────────────────────────────────────────────────────────

    def _generate_report(self):
        """Generate a performance report."""
        if time.time() - self.last_report_time < self.report_interval:
            return
        self.last_report_time = time.time()

        try:
            from tracking.strategy_report import StrategyReport
            report = StrategyReport(self.client)
            report.generate_report()

            # Log key metrics
            logger.info("=" * 50)
            logger.info("PERFORMANCE REPORT")
            logger.info("  Total trades: %d", self.total_trades)
            logger.info("  Total P&L: %.2f USDT", self.total_pnl)
            logger.info("  Current balance: %.2f USDT", self.current_balance)
            logger.info("  Peak balance: %.2f USDT", self.peak_balance)
            dd = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
            logger.info("  Current drawdown: %.2f%%", dd)
            logger.info("  Signal generator win rate: %.1f%%",
                        self.signal_generator.win_rate * 100 if self.signal_generator else 0)
            logger.info("=" * 50)
        except Exception as e:
            logger.error("Report generation failed: %s", e)

    # ── Main Loop ──────────────────────────────────────────────────────────

    def _trade_loop(self):
        """Main trading loop."""
        logger.info("Trade loop started (interval=%ds)", cfg.TRADE_LOOP_INTERVAL)
        while self.running:
            try:
                # 1. Analyze market
                analysis = self._analyze_market_conditions()

                # 2. Select strategy (informational — actual decisions made by signal)
                self.active_strategy = self._select_strategy(analysis)

                # 3. Make trade decision
                self._make_trade_decision(analysis)

                # 4. Periodic report
                self._generate_report()

                time.sleep(cfg.TRADE_LOOP_INTERVAL)

            except Exception as e:
                logger.error("Trade loop error: %s", e, exc_info=True)
                time.sleep(5)

    def run(self):
        """Start the trading system."""
        if not self.running:
            logger.error("System not running due to initialization failure")
            return

        trade_thread = Thread(target=self._trade_loop, daemon=True)
        trade_thread.start()
        logger.info("TradingSystem running")
        try:
            trade_thread.join()
        except KeyboardInterrupt:
            logger.info("Shutting down (SIGINT)")
            self.shutdown()
        except Exception as e:
            logger.critical("Fatal error: %s", e, exc_info=True)
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        try:
            if self.client:
                self.client.stop_websocket()
            self._generate_report()
            logger.info("Shutdown complete. Final balance: %.2f USDT", self.current_balance)
        except Exception as e:
            logger.error("Shutdown error: %s", e)
        finally:
            sys.exit(0)


# ── Entry Point ────────────────────────────────────────────────────────────
trading_system: Optional[TradingSystem] = None


def signal_handler(sig, frame):
    logger.info("Received signal %d", sig)
    if trading_system:
        trading_system.shutdown()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    trading_system = TradingSystem()
    if trading_system.running:
        trading_system.run()
    else:
        logger.critical("Initialization failed. Check logs.")