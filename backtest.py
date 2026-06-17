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

# We import the hedge fund strategy — same engine as live trading
from ai.self_learning import HedgeFundStrategy, TradeRecord, TradeSignal

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
        trade_size_btc: float = 0.001,
        max_position_btc: float = 0.1,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.trade_size = trade_size_btc
        self.max_position = max_position_btc

        self.strategy = HedgeFundStrategy()
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self.returns: List[float] = []

        self.position = {
            'size': 0.0, 'side': None, 'entry_price': 0.0,
            'stop_price': 0.0, 'target_price': 0.0,
            'highest_price': 0.0, 'lowest_price': 0.0,
            'regime': '',
        }
        self._current_idx = 0
        self._closes: np.ndarray = np.array([])

    def _calculate_position_size(self, price: float, atr: float) -> float:
        """Position sizing using ATR-based risk distance."""
        factor = self.strategy.get_position_sizing_factor()
        if factor <= 0.0:
            return 0.0

        stop_distance = atr * 1.5  # 1.5× ATR stop distance
        if stop_distance <= 0:
            stop_distance = price * 0.01
        risk_amount = self.balance * self.risk_per_trade
        size = risk_amount / stop_distance if stop_distance > 0 else self.trade_size
        size *= factor
        size = min(size, self.max_position)
        size = max(size, self.trade_size) if size > 0 else 0.0
        return size

    def _apply_slippage(self, price: float, is_buy: bool, slippage_bps: float = 2.0) -> float:
        """Apply slippage as fraction of price."""
        slippage = price * slippage_bps / 10000
        return price + slippage if is_buy else price - slippage

    def run(self, ohlcv: np.ndarray, verbose: bool = True) -> Dict:
        """
        Run backtest on historical data using the hedge fund strategy.

        Args:
            ohlcv: (N, 6) array [timestamp, open, high, low, close, volume].
            verbose: Print progress.

        Returns:
            Dict of performance metrics.
        """
        if ohlcv.shape[0] < 60:
            logger.error("Need at least 60 candles for backtest")
            return {}

        closes = ohlcv[:, 4]
        highs = ohlcv[:, 2]
        lows = ohlcv[:, 3]
        volumes = ohlcv[:, 5]
        self._closes = closes

        logger.info("Running backtest on %d candles (%.1f hours)", len(closes), len(closes))

        warmup = 60  # need enough data for EMAs and ATR

        for i in range(warmup, len(closes)):
            self._current_idx = i
            current_price = closes[i]

            # Build lookback arrays
            lookback_closes = closes[:i + 1]
            lookback_highs = highs[:i + 1]
            lookback_lows = lows[:i + 1]
            lookback_vols = volumes[:i + 1] if len(volumes) > i else np.ones(i + 1)

            # Track highest/lowest since entry for trailing stop
            if self.position['size'] > 0:
                if self.position['side'] == 'long':
                    self.position['highest_price'] = max(self.position['highest_price'], current_price)
                else:
                    self.position['lowest_price'] = min(self.position['lowest_price'], current_price)

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

            # ── Manage open position ──────────────────────────────────────
            if self.position['size'] > 0:
                self._manage_position(current_price, lows[i], highs[i])
                if self.position['size'] > 0:
                    # Check signal reversal (strong opposite signal)
                    rev_signal = self.strategy.generate_signal(
                        lookback_closes, lookback_highs, lookback_lows, lookback_vols
                    )
                    if self.position['side'] == 'long' and rev_signal.action == 'sell':
                        self._close_position(current_price, 'signal_reversal', rev_signal)
                    elif self.position['side'] == 'short' and rev_signal.action == 'buy':
                        self._close_position(current_price, 'signal_reversal', rev_signal)
                continue

            # ── No position — check for entry ─────────────────────────────
            signal = self.strategy.generate_signal(
                lookback_closes, lookback_highs, lookback_lows, lookback_vols
            )

            if signal.action in ('buy', 'sell') and signal.confidence >= 0.4:
                action = signal.action.upper()
                size = self._calculate_position_size(current_price, self.strategy._atr(
                    lookback_highs, lookback_lows, lookback_closes, 14
                )[-1])

                if size <= 0:
                    continue

                exec_price = self._apply_slippage(current_price, action == 'BUY')

                self.position = {
                    'size': size,
                    'side': action.lower(),
                    'entry_price': exec_price,
                    'stop_price': signal.stop_price,
                    'target_price': signal.target_price,
                    'highest_price': exec_price,
                    'lowest_price': exec_price,
                    'regime': signal.regime,
                }
                logger.info("OPEN %s %s: %.6f @ %.2f | SL=%.2f TP=%.2f | %s",
                            action, signal.regime, size, exec_price,
                            signal.stop_price, signal.target_price, signal.rationale)

        # Close any remaining position
        if self.position['size'] > 0:
            self._close_position(self._closes[-1], 'end_of_test')

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

    def _manage_position(self, close_price: float, low_price: float, high_price: float):
        """Manage open position — check stops, targets, and trailing stops.

        Uses the ATR-based stop/target prices stored at entry, and checks
        intra-candle extremes to prevent gap-through.
        """
        entry = self.position['entry_price']
        stop = self.position['stop_price']
        target = self.position['target_price']
        if entry <= 0 or stop <= 0:
            return

        if self.position['side'] == 'long':
            # Stop: check low against stop price (intra-candle)
            if low_price <= stop:
                # Fill at stop price — prevents gap-through where close is far below stop
                self._close_position(stop, 'stop_loss')
            # Target: check close against target
            elif close_price >= target:
                self._close_position(close_price, 'take_profit')
            # Trailing stop: update stop if trailing activation threshold met
            elif self.position['highest_price'] > entry:
                run_pct = (self.position['highest_price'] - entry) / entry
                if run_pct > 0.02:  # 2% profit → activate trailing
                    trail = self.position['highest_price'] - (entry * 0.005)
                    self.position['stop_price'] = max(self.position['stop_price'], trail)
                    if close_price <= self.position['stop_price']:
                        self._close_position(close_price, 'trailing_stop')
        else:
            # Short position
            if high_price >= stop:
                # Fill at stop price — prevents gap-through where close is far above stop
                self._close_position(stop, 'stop_loss')
            elif close_price <= target:
                self._close_position(close_price, 'take_profit')
            elif self.position['lowest_price'] < entry:
                run_pct = (entry - self.position['lowest_price']) / entry
                if run_pct > 0.02:
                    trail = self.position['lowest_price'] + (entry * 0.005)
                    self.position['stop_price'] = min(self.position['stop_price'], trail)
                    if close_price >= self.position['stop_price']:
                        self._close_position(close_price, 'trailing_stop')

    def _close_position(self, price: float, exit_reason: str, signal=None):
        """Close position and record trade."""
        side = self.position['side']
        size = self.position['size']
        entry = self.position['entry_price']
        regime = self.position.get('regime', signal.regime if signal else '')
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
            'reason': exit_reason, 'regime': regime,
            'candle_idx': self._current_idx,
        }
        self.trades.append(trade)

        logger.info("CLOSE %s %s: %.4f USDT (%.2f%%) [%s]", side.upper(), regime, pnl, pnl_pct, exit_reason)

        # Reset position
        self.position = {
            'size': 0.0, 'side': None, 'entry_price': 0.0,
            'stop_price': 0.0, 'target_price': 0.0,
            'highest_price': 0.0, 'lowest_price': 0.0,
            'regime': '',
        }

        # Feed back to strategy for win/loss tracking
        trade_record = TradeRecord(
            entry_price=entry, exit_price=exec_price,
            side=side, size=size, pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=exit_reason, regime=regime,
            timestamp=time.time()
        )
        self.strategy.record_trade_outcome(trade_record)

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
            testnet=True  # Use testnet (matches current API key permissions)
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