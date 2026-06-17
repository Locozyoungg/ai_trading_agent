"""
Order Book Analysis Module

Analyzes order book data from BybitClient to compute trading signals.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OrderBookAnalysis:
    """
    Analyzes order book data to calculate market pressure indicators.
    """

    def __init__(self, client, symbol: str = "BTCUSDT"):
        """
        Args:
            client: BybitClient instance (not BybitAPI).
            symbol: Trading pair.
        """
        self.client = client
        self.symbol = symbol

    def calculate_order_flow_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate normalized Order Flow Imbalance from order book.

        Positive = buy pressure, negative = sell pressure.
        Range: [-1.0, 1.0]

        Args:
            levels: Number of top bid/ask levels to consider.

        Returns:
            Normalized OFI or None if data unavailable.
        """
        try:
            data = self.client.get_order_book(self.symbol)
            if not data or 'b' not in data or 'a' not in data:
                logger.warning("Invalid order book data for %s", self.symbol)
                return None

            bids = data['b'][:levels]
            asks = data['a'][:levels]

            if not bids or not asks:
                return None

            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            total = bid_vol + ask_vol

            if total == 0:
                return 0.0

            ofi = (bid_vol - ask_vol) / total
            logger.debug("OFI=%+.4f (bid_vol=%.2f ask_vol=%.2f)", ofi, bid_vol, ask_vol)
            return ofi

        except Exception as e:
            logger.error("OFI calculation error: %s", e, exc_info=True)
            return None

    def calculate_spread_pct(self) -> Optional[float]:
        """Calculate current bid-ask spread as percentage of mid-price."""
        try:
            data = self.client.get_order_book(self.symbol)
            if not data or 'b' not in data or 'a' not in data:
                return None
            best_bid = float(data['b'][0][0])
            best_ask = float(data['a'][0][0])
            if best_bid <= 0:
                return None
            return (best_ask - best_bid) / best_bid * 100
        except Exception as e:
            logger.error("Spread calc error: %s", e)
            return None

    def calculate_bid_ask_ratio(self, levels: int = 5) -> Optional[float]:
        """Ratio of total bid volume to total ask volume (>1 = buy pressure)."""
        try:
            data = self.client.get_order_book(self.symbol)
            if not data or 'b' not in data or 'a' not in data:
                return None
            bid_vol = sum(float(b[1]) for b in data['b'][:levels])
            ask_vol = sum(float(a[1]) for a in data['a'][:levels])
            if ask_vol == 0:
                return 2.0  # extreme buy pressure
            return bid_vol / ask_vol
        except Exception as e:
            logger.error("Bid/ask ratio error: %s", e)
            return None


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
    oba = OrderBookAnalysis(client, "BTCUSDT")
    print("OFI:", oba.calculate_order_flow_imbalance())
    print("Spread %:", oba.calculate_spread_pct())
    print("Bid/Ask Ratio:", oba.calculate_bid_ask_ratio())