"""
Historical Backtester for the AI Trading Agent

Simulates the trading strategy on historical OHLCV data and reports
performance metrics: total return, Sharpe ratio, max drawdown, win rate.

Usage:
    python backtest.py --symbol BTCUSDT --days 30
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('backtest')

# We import the signal generator to use the same logic as live trading
from ai.self_learning import SignalGenerator, TradeRecord

# ── Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(equity_curve: np.ndarray, daily_returns: np.ndarray) -> Dict:
    """
    Compute standard trading performance metrics.

    Args:
        equity_curve: Array of portfolio values over time.
        daily_returns: Array of daily return percentages.

    Returns:
        Dict with 'total_return', 'annual_return', 'sharpe_ratio',
        'max_drawdown', 'win_rate', 'total_trades', 'avg_win', 'avg_loss'.
    """
    if len(equity_curve) < 2:
        return {'error': 'Insufficient data'}

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Annualized return (assume 365 'candles' per year for 1h data)
    n_candles = len(equity_curve)
    ann_return = (1 + total_return) ** (365 * 24 / n_candles) - 1 if n_candles > 0 else 0.0

    # Sharpe ratio (risk-free rate = 0 for simplicity)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365 * 24)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    return {
        'total_return': float(total_return),
        'ann_return': float(ann_return),
        'sharpe_ratio': round(float(sharpe), 3),
        'max_drawdown': float(max_dd),
    }


def compute_trade_metrics(trades: List[Dict]) -> Dict:
    """Compute metrics from a list of completed trades."""
    if not trades:
        return {'total_trades': 0, 'win_rate': 0.0}

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 4),
        'avg_win_pct': round(float(avg_win), 4),
        'avg_loss_pct': round(float(avg_loss), 4),
        'profit_factor': round(float(profit_factor), 3),
        'total_pnl': round(sum(pnls), 4),
    }


# ── Backtest Engine ────────────────────────────────────────────────────────


class BacktestEngine:
    """
    Simple but rigorous backtester.

    Walks through historical OHLCV data, generates signals at each step,
    simulates fills at candle close, and tracks equity.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.03,
        trade_size_btc: float = 0.001,
        max_position_btc: float = 0.1,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trade_size = trade_size_btc
        self.max_position = max_position_btc

        self.signal_gen = SignalGenerator()
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self.returns: List[float] = []

        self.position = {'size': 0.0, 'side': None, 'entry_price': 0.0}
        self._current_idx = 0
        self._prices: np.ndarray = np.array([])

    def _calculate_position_size(self, price: float) -> float:
        """Simple fixed-fraction position sizing."""
        risk_amount = self.balance * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        size = risk_amount / stop_distance if stop_distance > 0 else self.trade_size
        size = min(size, self.max_position)
        size = max(size, self.trade_size)
        return size

    def _apply_slippage(self, price: float, is_buy: bool, slippage_bps: float = 2.0) -> float:
        """Apply slippage as fraction of price."""
        slippage = price * slippage_bps / 10000
        return price + slippage if is_buy else price - slippage

    def run(self, ohlcv: np.ndarray, verbose: bool = True) -> Dict:
        """
        Run backtest on historical data.

        Args:
            ohlcv: (N, 6) array [timestamp, open, high, low, close, volume].
            verbose: Print progress.

        Returns:
            Dict of performance metrics.
        """
        if ohlcv.shape[0] < 50:
            logger.error("Need at least 50 candles for backtest")
            return {}

        self._prices = ohlcv[:, 4]  # close prices
        volumes = ohlcv[:, 5]

        logger.info("Running backtest on %d candles (%.1f hours)",
                    len(self._prices), len(self._prices))

        for i in range(50, len(self._prices)):
            self._current_idx = i
            current_price = self._prices[i]
            lookback_prices = self._prices[:i + 1]
            lookback_volumes = volumes[:i + 1] if len(volumes) > i else np.ones(i + 1)

            # Update equity
            unrealized = 0.0
            if self.position['size'] > 0:
                if self.position['side'] == 'long':
                    unrealized = (current_price - self.position['entry_price']) * self.position['size']
                else:
                    unrealized = (self.position['entry_price'] - current_price) * self.position['size']
            equity = self.balance + unrealized
            self.equity_curve.append(equity)
            if len(self.equity_curve) > 1:
                self.returns.append((self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2])

            # Manage open position
            if self.position['size'] > 0:
                self._manage_position(current_price)
                if self.position['size'] > 0:
                    # Check signal reversal
                    signal = self.signal_gen.generate_signal(
                        lookback_prices, lookback_volumes
                    )
                    if self.position['side'] == 'long' and signal.strength < -0.4:
                        self._close_position(current_price, 'signal_reversal')
                    elif self.position['side'] == 'short' and signal.strength > 0.4:
                        self._close_position(current_price, 'signal_reversal')
                continue

            # No position — check for entry
            signal = self.signal_gen.generate_signal(
                lookback_prices, lookback_volumes
            )

            if signal.action in ('buy', 'sell') and signal.confidence >= 0.35:
                action = signal.action.upper()
                size = self._calculate_position_size(current_price)
                exec_price = self._apply_slippage(current_price, action == 'BUY')

                self.position = {
                    'size': size,
                    'side': action.lower(),
                    'entry_price': exec_price,
                }
                logger.debug("Candle %d: OPEN %s %.6f @ %.2f (conf=%.3f, regime=%s)",
                             i, action, size, exec_price, signal.confidence, signal.regime.name)

        # Close any remaining position
        if self.position['size'] > 0:
            self._close_position(self._prices[-1], 'end_of_test')

        # Compute and return metrics
        equity_arr = np.array(self.equity_curve)
        returns_arr = np.array(self.returns)

        metrics = compute_metrics(equity_arr, returns_arr)
        trade_metrics = compute_trade_metrics(self.trades)
        metrics.update(trade_metrics)

        metrics['final_balance'] = round(self.balance, 2)
        metrics['total_return_pct'] = round(metrics.get('total_return', 0) * 100, 2)
        metrics['ann_return_pct'] = round(metrics.get('ann_return', 0) * 100, 2)
        metrics['max_drawdown_pct'] = round(metrics.get('max_drawdown', 0) * 100, 2)

        if verbose:
            self._print_summary(metrics)

        return metrics

    def _manage_position(self, current_price: float):
        """Apply stop-loss and take-profit."""
        entry = self.position['entry_price']
        if entry <= 0:
            return

        if self.position['side'] == 'long':
            pnl_pct = (current_price - entry) / entry
            if pnl_pct <= -self.stop_loss_pct:
                self._close_position(current_price, 'stop_loss')
            elif pnl_pct >= self.take_profit_pct:
                self._close_position(current_price, 'take_profit')
        else:
            pnl_pct = (entry - current_price) / entry
            if pnl_pct <= -self.stop_loss_pct:
                self._close_position(current_price, 'stop_loss')
            elif pnl_pct >= self.take_profit_pct:
                self._close_position(current_price, 'take_profit')

    def _close_position(self, price: float, reason: str):
        """Close position and record trade."""
        side = self.position['side']
        size = self.position['size']
        entry = self.position['entry_price']
        exec_price = self._apply_slippage(price, side == 'sell')

        if side == 'long':
            pnl = (exec_price - entry) * size
            pnl_pct = (exec_price - entry) / entry * 100
        else:
            pnl = (entry - exec_price) * size
            pnl_pct = (entry - exec_price) / entry * 100

        self.balance += pnl

        trade = {
            'entry_price': entry, 'exit_price': exec_price,
            'side': side, 'size': size,
            'pnl': pnl, 'pnl_pct': pnl_pct,
            'reason': reason, 'candle_idx': self._current_idx,
        }
        self.trades.append(trade)

        logger.debug("CLOSE %s @ %.2f | PnL: %.2f (%.2f%%) [%s]",
                     side.upper(), exec_price, pnl, pnl_pct, reason)

        self.position = {'size': 0.0, 'side': None, 'entry_price': 0.0}

        # Feed back to signal generator
        trade_record = TradeRecord(
            entry_price=entry, exit_price=exec_price,
            side=side, size=size, pnl=pnl, pnl_pct=pnl_pct,
            signal_strength=0.0, regime='',
            timestamp=time.time()
        )
        self.signal_gen.record_trade_outcome(trade_record)

    def _print_summary(self, metrics: Dict):
        """Pretty-print backtest results."""
        print("\n" + "=" * 55)
        print("  BACKTEST RESULTS")
        print("=" * 55)
        print(f"  Initial balance:  ${self.initial_balance:.2f}")
        print(f"  Final balance:    ${metrics.get('final_balance', 0):.2f}")
        print(f"  Total return:     {metrics.get('total_return_pct', 0):+.2f}%")
        print(f"  Annualized return: {metrics.get('ann_return_pct', 0):+.2f}%")
        print(f"  Sharpe ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Total trades:     {metrics.get('total_trades', 0)}")
        print(f"  Win rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Profit factor:    {metrics.get('profit_factor', 0):.2f}")
        print(f"  Avg win:          {metrics.get('avg_win_pct', 0):+.2f}%")
        print(f"  Avg loss:         {metrics.get('avg_loss_pct', 0):.2f}%")
        print("=" * 55)


# ── Data Fetcher ───────────────────────────────────────────────────────────


def fetch_historical_data(symbol: str, days: int = 30) -> Optional[np.ndarray]:
    """Fetch historical OHLCV data from Bybit via BybitClient."""
    try:
        from bybit_client import BybitClient
        from dotenv import load_dotenv
        import os
        load_dotenv()

        client = BybitClient(
            os.getenv("BYBIT_API_KEY", ""),
            os.getenv("BYBIT_API_SECRET", ""),
            testnet=False  # Use mainnet for historical data
        )

        # Fetch in batches (max 200 per request)
        all_data = []
        limit = 200
        total_needed = days * 24  # ~1h candles
        fetched = 0

        while fetched < total_needed:
            ohlcv = client.get_historical_data(symbol, interval='60', limit=limit)
            if ohlcv is None or len(ohlcv) == 0:
                break
            if len(all_data) > 0 and np.array_equal(ohlcv[-1], all_data[0]):
                break
            all_data = list(ohlcv) + all_data
            fetched += len(ohlcv)
            if len(ohlcv) < limit:
                break
            time.sleep(0.5)

        if all_data:
            return np.array(all_data)
        return None

    except Exception as e:
        logger.error("Failed to fetch historical data: %s", e)
        return None


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Backtest the AI Trading Agent')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=14, help='Days of data')
    parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade')
    parser.add_argument('--file', type=str, help='Path to CSV data file (optional)')

    args = parser.parse_args()

    # Load data
    if args.file:
        df = pd.read_csv(args.file)
        if 'close' in df.columns:
            prices = df['close'].values
            volumes = df.get('volume', pd.Series(np.ones(len(prices)))).values
            ohlcv = np.column_stack([
                np.arange(len(prices)),  # timestamp placeholder
                prices, prices, prices, prices, volumes
            ])
        else:
            logger.error("CSV must have a 'close' column")
            sys.exit(1)
        logger.info("Loaded %d candles from %s", len(ohlcv), args.file)
    else:
        logger.info("Fetching %d days of %s data...", args.days, args.symbol)
        ohlcv = fetch_historical_data(args.symbol, args.days)
        if ohlcv is None or len(ohlcv) < 50:
            logger.error("Insufficient historical data")
            sys.exit(1)
        logger.info("Fetched %d candles", len(ohlcv))

    # Run backtest
    engine = BacktestEngine(
        initial_balance=args.balance,
        risk_per_trade=args.risk,
    )
    metrics = engine.run(ohlcv, verbose=True)

    # Save results
    from config import BACKTEST_RESULTS_DIR
    results_file = BACKTEST_RESULTS_DIR / f"backtest_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    if engine.trades:
        df_trades = pd.DataFrame(engine.trades)
        df_trades.to_csv(results_file, index=False)
        logger.info("Trade log saved to %s", results_file)

    # Save equity curve
    equity_file = BACKTEST_RESULTS_DIR / f"equity_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame({'equity': engine.equity_curve}).to_csv(equity_file, index=False)
    logger.info("Equity curve saved to %s", equity_file)

    return metrics


if __name__ == "__main__":
    metrics = main()