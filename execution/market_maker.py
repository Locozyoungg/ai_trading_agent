"""
Market Making Execution Module

Places bid and ask limit orders around the mid-price to capture the spread.
Includes order tracking, cancellation of stale orders, and inventory management.
"""

import logging
import time
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class MarketMaker:
    """
    Market-making strategy that maintains two-sided limit orders.

    Manages its own order tracking to avoid leaving stale orders on the book.
    Pairs with risk management to limit inventory exposure.
    """

    def __init__(
        self,
        client,
        symbol: str = "BTCUSDT",
        spread: float = 0.0005,
        size: float = 0.01,
        position_info: Optional[dict] = None,
        risk_components: Optional[dict] = None,
        max_inventory_ratio: float = 0.5,
    ):
        """
        Args:
            client: BybitClient instance.
            symbol: Trading pair.
            spread: Target half-spread as fraction of mid-price.
            size: Base order size in BTC.
            position_info: Shared position info.
            risk_components: Shared risk components.
            max_inventory_ratio: Max net inventory relative to base size.
        """
        self.client = client
        self.symbol = symbol
        self.spread = spread
        self.base_size = size
        self.position_info = position_info or {}
        self.risk_components = risk_components or {}
        self.max_inventory_ratio = max_inventory_ratio

        # Track our own orders so we can cancel stale ones
        self.open_order_ids: List[str] = []
        self.running = False
        logger.info(
            "MarketMaker initialized for %s: spread=%.4f%% size=%.4f",
            symbol, spread * 200, size
        )

    def cancel_all_orders(self):
        """Cancel all tracked open orders."""
        for order_id in list(self.open_order_ids):
            try:
                self.client.cancel_order(self.symbol, order_id)
                logger.debug("Cancelled order %s", order_id)
            except Exception as e:
                logger.debug("Order %s already filled/cancelled: %s", order_id, e)
        self.open_order_ids.clear()

    def execute_market_making(self):
        """
        Place bid and ask orders around the mid-price.

        1. Cancel existing orders
        2. Check inventory limits
        3. Place new bid and ask orders
        """
        # Check if we already have a directional position
        current_position = self.position_info.get('size', 0)
        if abs(current_position) > self.base_size * self.max_inventory_ratio:
            logger.debug("Inventory limit reached (%f). Skipping market making.", current_position)
            self.cancel_all_orders()
            return

        try:
            book = self.client.get_order_book(self.symbol)
            if not book or 'b' not in book or 'a' not in book:
                return

            best_bid = float(book['b'][0][0]) if book['b'] else None
            best_ask = float(book['a'][0][0]) if book['a'] else None

            if not best_bid or not best_ask:
                return

            mid = (best_bid + best_ask) / 2

            # Cancel old orders before placing new ones
            self.cancel_all_orders()

            # Apply position sizing from risk management if available
            size = self.base_size
            if 'position_sizing' in self.risk_components:
                try:
                    size = self.risk_components['position_sizing'].calculate_position_size(size)
                except Exception:
                    pass

            bid_price = round(mid * (1 - self.spread), 2)
            ask_price = round(mid * (1 + self.spread), 2)

            # Place bid
            bid_order = self.client.place_order(
                self.symbol, size, "BUY", order_type="Limit", price=bid_price
            )
            if bid_order and 'id' in bid_order:
                self.open_order_ids.append(bid_order['id'])

            # Place ask
            ask_order = self.client.place_order(
                self.symbol, size, "SELL", order_type="Limit", price=ask_price
            )
            if ask_order and 'id' in ask_order:
                self.open_order_ids.append(ask_order['id'])

            logger.info(
                "Market making: bid=%.2f ask=%.2f mid=%.2f spread=%.4f%%",
                bid_price, ask_price, mid, (ask_price - bid_price) / mid * 100
            )

        except Exception as e:
            logger.error("Market making error: %s", e, exc_info=True)

    def stop(self):
        """Cancel orders and stop."""
        self.running = False
        self.cancel_all_orders()
        logger.info("MarketMaker stopped for %s", self.symbol)


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
    mm = MarketMaker(client, "BTCUSDT", spread=0.0005, size=0.001)
    mm.execute_market_making()
    mm.stop()