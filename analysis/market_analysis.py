"""
Market Insights Module

Fetches and analyzes OHLCV market data through BybitClient to generate
trading insights including volatility, trend direction, and support/resistance.
"""

import logging
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class MarketInsights:
    """
    Provides market analysis based on OHLCV data fetched via BybitClient.
    """

    def __init__(self, client, symbols: List[str], timeframe: str = "1h"):
        """
        Args:
            client: BybitClient instance.
            symbols: List of trading symbols.
            timeframe: Candle interval (e.g. '1', '5', '15', '60', 'D').
        """
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe

    def get_latest_data(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Fetch OHLCV data as numpy arrays via BybitClient.

        Returns:
            dict: symbol → (N, 6) ndarray [timestamp, O, H, L, C, V] or None.
        """
        result = {}
        for symbol in self.symbols:
            try:
                data = self.client.get_historical_data(
                    symbol,
                    interval=self.timeframe,
                    limit=60
                )
                # get_historical_data returns (N,6) ndarray
                if data is not None and len(data) > 0:
                    result[symbol] = data
                    logger.info("Fetched %d candles for %s", len(data), symbol)
                else:
                    logger.warning("No OHLCV data for %s", symbol)
                    result[symbol] = None
            except Exception as e:
                logger.error("Error fetching data for %s: %s", symbol, e)
                result[symbol] = None
        return result

    def analyze_market(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze market conditions and return trade insights per symbol.

        Returns dict with keys:
            entry_price, stop_loss_pct, take_profit_pct, avg_close,
            volatility, trend_strength, rsi_approximation
        """
        data = self.get_latest_data()
        insights = {}

        for symbol, candles in data.items():
            if candles is None or len(candles) < 20:
                insights[symbol] = {
                    "entry_price": None,
                    "stop_loss_pct": None,
                    "take_profit_pct": None,
                    "avg_close": None,
                    "volatility": None,
                    "trend_strength": 0.0,
                }
                continue

            try:
                # Column layout: [timestamp, open, high, low, close, volume]
                closes = candles[:, 4]
                highs = candles[:, 2]
                lows = candles[:, 3]
                latest_close = closes[-1]

                # Volatility: standard deviation of daily returns
                returns = np.diff(closes) / closes[:-1]
                volatility = float(np.std(returns)) if len(returns) > 0 else 0.0

                # Trend strength via linear regression slope
                x = np.arange(len(closes))
                slope = np.polyfit(x, closes, 1)[0]
                trend_strength = np.tanh(slope / (closes.mean() + 1e-10) * 100)

                # Simple RSI approximation (14-period)
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                rsi = 50.0
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))

                # Adaptive stop-loss based on volatility
                atr = float(np.mean(highs[-14:] - lows[-14:])) if len(highs) >= 14 else latest_close * 0.01
                stop_loss_pct = min(5.0, max(0.5, atr / latest_close * 100 * 1.5))
                take_profit_pct = stop_loss_pct * 2.0  # 2:1 risk-reward

                insights[symbol] = {
                    "entry_price": float(latest_close),
                    "stop_loss_pct": round(stop_loss_pct, 2),
                    "take_profit_pct": round(take_profit_pct, 2),
                    "avg_close": float(np.mean(closes)),
                    "volatility": round(volatility * 100, 3),
                    "trend_strength": round(trend_strength, 4),
                    "rsi": round(rsi, 1),
                }
                logger.info(
                    "Analysis %s: close=%.2f vol=%.3f%% trend=%.3f rsi=%.1f",
                    symbol, latest_close, volatility * 100, trend_strength, rsi
                )

            except Exception as e:
                logger.error("Analysis failed for %s: %s", symbol, e)
                insights[symbol] = {"entry_price": None, "stop_loss_pct": None}

        return insights


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from bybit_client import BybitClient

    client = BybitClient(
        os.getenv("BYBIT_API_KEY", ""),
        os.getenv("BYBIT_API_SECRET", ""),
        testnet=True
    )
    mi = MarketInsights(client, ["BTCUSDT", "ETHUSDT"], timeframe="60")
    result = mi.analyze_market()
    for sym, data in result.items():
        print(f"\n{sym}:")
        for k, v in data.items():
            print(f"  {k}: {v}")