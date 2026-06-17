"""
Stop Hunt Detector

Detects potential stop-hunting activity by identifying rapid,
unusual price movements beyond normal volatility thresholds.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class StopHuntDetector:
    """
    Monitors price action for signs of stop hunts — sharp moves
    designed to trigger clustered stop-loss orders before reversing.
    """

    def __init__(self, client, symbol: str = "BTCUSDT", lookback_period: int = 60):
        """
        Args:
            client: BybitClient instance.
            symbol: Trading pair.
            lookback_period: Seconds of data to analyze.
        """
        self.client = client
        self.symbol = symbol
        self.lookback_period = lookback_period

    def detect_stop_hunts(self, recent_data: pd.DataFrame, threshold: float = 0.005) -> bool:
        """
        Detect stop hunts — sharp price reversals after a rapid move.

        A stop hunt is flagged when price moves beyond *threshold*
        and then reverses >50% of the move within the next 3 candles.

        Args:
            recent_data: DataFrame with 'close' column, sorted chronologically.
            threshold: Min price change (%) to consider as potential hunt.

        Returns:
            True if stop hunt pattern detected.
        """
        try:
            if not isinstance(recent_data, pd.DataFrame):
                logger.error("Input must be a DataFrame, got %s", type(recent_data))
                return False

            if 'close' not in recent_data.columns:
                logger.error("Missing 'close' column")
                return False

            prices = pd.to_numeric(recent_data['close'], errors='coerce').dropna().values
            if len(prices) < 6:
                return False

            returns = np.diff(prices) / prices[:-1]

            for i in range(2, len(returns) - 3):
                # Sharp move
                if abs(returns[i]) > threshold:
                    move_dir = np.sign(returns[i])
                    # Check for reversal in next 3 candles
                    reversal = returns[i + 1:i + 4]
                    if len(reversal) > 0:
                        avg_reversal = np.mean(reversal)
                        # Reversed >50% of initial move
                        if move_dir * avg_reversal < -0.5 * abs(returns[i]):
                            logger.info(
                                "Stop hunt detected: %.4f move reversed at index %d",
                                returns[i], i
                            )
                            return True

            logger.debug("No stop hunt pattern found")
            return False

        except Exception as e:
            logger.error("Stop hunt detection failed: %s", e, exc_info=True)
            return False


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
    detector = StopHuntDetector(client, "BTCUSDT", lookback_period=60)

    # Fetch sample data
    ohlcv = client.get_historical_data("BTCUSDT", interval="1", limit=30)
    if ohlcv is not None and len(ohlcv) > 0:
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = detector.detect_stop_hunts(df)
        print("Stop hunt detected:", result)
    else:
        print("Failed to fetch sample data")