"""
Iceberg Order Detector

Detects potential iceberg (hidden) orders by monitoring order book
for repetitive refresh patterns at the same price level.
"""

import time
import logging
from collections import defaultdict
from typing import List

logger = logging.getLogger(__name__)


class IcebergDetector:
    """
    Detects iceberg orders by monitoring order book snapshots over time.
    An iceberg order is identified when the same price level shows
    multiple distinct size values across consecutive polls, suggesting
    a large hidden order being fed in small chunks.
    """

    def __init__(self, client, symbol: str = "BTCUSDT"):
        """
        Args:
            client: BybitClient instance.
            symbol: Trading pair.
        """
        self.client = client
        self.symbol = symbol

    def detect_iceberg_orders(self, cycles: int = 10, refresh_threshold: int = 3) -> List[float]:
        """
        Poll the order book and flag price levels with size variations.

        Args:
            cycles: Number of sequential order-book snapshots to compare.
            refresh_threshold: Minimum distinct size observations to flag.

        Returns:
            List of price levels where iceberg activity is suspected.
        """
        iceberg_data = defaultdict(list)  # price → [sizes]

        for i in range(cycles):
            try:
                data = self.client.get_order_book(self.symbol)
                if not data or 'b' not in data or 'a' not in data:
                    logger.warning("Order book unavailable on cycle %d", i)
                    time.sleep(1)
                    continue

                for level in data['b'][:10] + data['a'][:10]:
                    price = float(level[0])
                    size = float(level[1])
                    iceberg_data[price].append(size)

            except Exception as e:
                logger.error("Iceberg detection cycle %d failed: %s", i, e)

            time.sleep(1)  # pause between polls

        # Flag prices where size varied significantly across cycles
        detected = [
            price for price, sizes in iceberg_data.items()
            if len(set(round(s, 6) for s in sizes)) >= refresh_threshold
        ]

        if detected:
            logger.info("Potential iceberg orders at: %s", detected)
        else:
            logger.debug("No iceberg patterns detected")

        return detected


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
    detector = IcebergDetector(client, "BTCUSDT")
    result = detector.detect_iceberg_orders(cycles=5)
    print("Detected icebergs:", result)