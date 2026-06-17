"""
High-Frequency Trading Execution Module

Executes rapid trades based on order-book imbalances and spreads.
Uses the centralized BybitClient WebSocket data — no duplicate connections.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class HFTTrading:
    """
    High-frequency trading strategy that exploits order-book imbalances.

    Does NOT open its own WebSocket — it reads from the shared BybitClient
    order book buffer to avoid duplicate connections and state conflicts.
    """

    def __init__(
        self,
        client,
        symbol: str = "BTCUSDT",
        position_info: Optional[dict] = None,
        risk_components: Optional[dict] = None,
        spread_threshold: float = 0.0002,
        order_size: float = 0.001,
    ):
        """
        Args:
            client: BybitClient instance (shared).
            symbol: Trading pair.
            position_info: Shared position dict from TradingSystem.
            risk_components: Shared risk components dict.
            spread_threshold: Min spread (%) to attempt a trade.
            order_size: BTC per HFT order.
        """
        self.client = client
        self.symbol = symbol
        self.position_info = position_info or {}
        self.risk_components = risk_components or {}
        self.spread_threshold = spread_threshold
        self.order_size = order_size
        self.running = False
        self._last_trade_time = 0
        logger.info("HFT initialized for %s (spread >= %.4f%%)", symbol, spread_threshold * 100)

    def execute_hft(self) -> Optional[str]:
        """
        Check order book for HFT opportunity and execute if favorable.

        Strategy: When spread is wide and OFI shows directional pressure,
        trade in the direction of the pressure.

        Returns:
            'buy', 'sell', or None if no trade executed.
        """
        # Only trade if no existing position
        if self.position_info.get('size', 0) > 0:
            return None

        # Rate limit: max one HFT trade per 5 seconds
        if time.time() - self._last_trade_time < 5:
            return None

        try:
            book = self.client.get_order_book(self.symbol)
            if not book or 'b' not in book or 'a' not in book:
                return None

            bids = book['b'][:5]
            asks = book['a'][:5]
            if not bids or not asks:
                return None

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread_pct = (best_ask - best_bid) / best_bid

            if spread_pct < self.spread_threshold:
                logger.debug("HFT: spread %.4f%% below threshold", spread_pct * 100)
                return None

            # Calculate pressure: bid volume vs ask volume at top levels
            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            total = bid_vol + ask_vol
            if total == 0:
                return None

            pressure = (bid_vol - ask_vol) / total  # -1 to 1

            # Only trade with conviction
            if pressure > 0.3:
                logger.info("HFT BUY signal: spread=%.4f%% pressure=%.3f", spread_pct * 100, pressure)
                order = self.client.place_order(self.symbol, self.order_size, "BUY", order_type="Market")
                if order and 'id' in order:
                    self._last_trade_time = time.time()
                    return "buy"
            elif pressure < -0.3:
                logger.info("HFT SELL signal: spread=%.4f%% pressure=%.3f", spread_pct * 100, pressure)
                order = self.client.place_order(self.symbol, self.order_size, "SELL", order_type="Market")
                if order and 'id' in order:
                    self._last_trade_time = time.time()
                    return "sell"

            return None

        except Exception as e:
            logger.error("HFT execution error: %s", e, exc_info=True)
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
    hft = HFTTrading(client, "BTCUSDT")
    result = hft.execute_hft()
    print("HFT result:", result)