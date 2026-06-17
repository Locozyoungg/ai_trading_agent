"""
Scalping Strategy Execution Module

Captures small, rapid profits by exploiting short-term order book imbalances.
No duplicate WebSockets — uses shared client data.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScalpingStrategy:
    """
    Scalping strategy: takes small, quick profits from order book pressure waves.

    Unlike HFT (which looks for spread opportunities), scalping focuses on
    rapid order-book changes and small price dislocations.
    """

    def __init__(
        self,
        client,
        symbol: str = "BTCUSDT",
        spread: float = 0.0002,
        size: float = 0.001,
        position_info: Optional[dict] = None,
        risk_components: Optional[dict] = None,
        min_confidence: float = 0.4,
    ):
        """
        Args:
            client: BybitClient instance.
            symbol: Trading pair.
            spread: Target half-spread as fraction.
            size: Base order size in BTC.
            position_info: Shared position info.
            risk_components: Shared risk components.
            min_confidence: Minimum signal confidence to scalp.
        """
        self.client = client
        self.symbol = symbol
        self.spread = spread
        self.base_size = size
        self.position_info = position_info or {}
        self.risk_components = risk_components or {}
        self.min_confidence = min_confidence
        logger.info("ScalpingStrategy initialized for %s", symbol)

    def execute_scalp(self) -> Optional[str]:
        """
        Look for a scalping opportunity.

        Strategy:
        - Check order book pressure (bid volume / ask volume ratio)
        - If strong buy pressure and price hasn't moved yet, buy for a quick scalp
        - If strong sell pressure and price hasn't moved yet, sell for a quick scalp
        - Target: capture 50% of the spread

        Returns:
            'buy', 'sell', or None.
        """
        if self.position_info.get('size', 0) > 0:
            return None

        try:
            book = self.client.get_order_book(self.symbol)
            if not book or 'b' not in book or 'a' not in book:
                return None

            bids = book['b'][:10]
            asks = book['a'][:10]
            if not bids or not asks:
                return None

            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            total = bid_vol + ask_vol

            if total == 0:
                return None

            ratio = bid_vol / ask_vol if ask_vol > 0 else 2.0

            # Strong buy pressure (ratio > 1.5)
            if ratio > 1.5:
                logger.debug("Scalp BUY opportunity: bid/ask ratio=%.3f", ratio)
                size = self.base_size
                order = self.client.place_order(self.symbol, size, "BUY", order_type="Market")
                if order and 'id' in order:
                    logger.info("Scalp BUY executed: %.4f @ market", size)
                    return "buy"

            # Strong sell pressure (ratio < 0.67)
            elif ratio < 0.67:
                logger.debug("Scalp SELL opportunity: bid/ask ratio=%.3f", ratio)
                size = self.base_size
                order = self.client.place_order(self.symbol, size, "SELL", order_type="Market")
                if order and 'id' in order:
                    logger.info("Scalp SELL executed: %.4f @ market", size)
                    return "sell"

            return None

        except Exception as e:
            logger.error("Scalp execution error: %s", e, exc_info=True)
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
    scalper = ScalpingStrategy(client, "BTCUSDT")
    result = scalper.execute_scalp()
    print("Scalp result:", result)