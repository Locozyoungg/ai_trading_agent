# data_pipeline/bybit_api.py
"""
Bybit API Interface Module

Provides methods to interact with Bybit's API for trading and data retrieval
"""

import time
import logging
from pybit.unified_trading import HTTP

# Set up logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize BybitAPI with API credentials.

        Args:
            api_key (str): Bybit API key
            api_secret (str): Bybit API secret
            testnet (bool): Use testnet if True, mainnet if False (default: True)
        """
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("BybitAPI initialized")

    def get_btc_price(self) -> float | None:
        """
        Fetch the latest BTC price.

        Returns:
            float | None: Last price of BTCUSDT if successful, None otherwise
        """
        try:
            response = self.client.get_tickers(category="linear", symbol="BTCUSDT")
            if response.get("retCode") != 0:
                logger.error(f"API error fetching BTC price: {response.get('retMsg')}")
                return None
            price = float(response["result"]["list"][0]["lastPrice"])
            logger.info(f"BTC Price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching BTC price: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 10) -> list:
        """
        Fetch recent trading records for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            limit (int): Number of trades to fetch (default: 10)

        Returns:
            list: List of recent trades if successful, empty list otherwise
        """
        try:
            response = self.client.get_public_trade(
                category="linear",
                symbol=symbol,
                limit=limit
            )
            if response.get("retCode") != 0:
                logger.error(f"API error fetching recent trades for {symbol}: {response.get('retMsg')}")
                return []
            trades = response.get("result", {}).get("list", [])
            logger.info(f"Fetched {len(trades)} recent trades for {symbol}")
            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return []

    def get_open_positions(self, symbol: str) -> list:
        """
        Fetch open positions for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            list: List of position details if successful, empty list otherwise
        """
        try:
            response = self.client.get_positions(category="linear", symbol=symbol)
            if response.get("retCode") != 0:
                logger.error(f"API error fetching positions for {symbol}: {response.get('retMsg')}")
                return []
            positions = response["result"]["list"]
            logger.info(f"Fetched {len(positions)} open positions for {symbol}")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions for {symbol}: {e}")
            return []

    def close_position(self, symbol: str):
        """
        Close an open position for a given symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
        """
        positions = self.get_open_positions(symbol)
        position = next((p for p in positions if p["symbol"] == symbol and float(p["size"]) > 0), None)

        if position:
            try:
                close_order = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side="Sell" if position["side"] == "Buy" else "Buy",
                    orderType="Market",
                    qty=str(position["size"]),
                    reduceOnly=True
                )
                logger.info(f"Close Position Response: {close_order}")
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
        else:
            logger.info(f"No open position to close for {symbol}")

    def place_order(self, symbol: str, side: str, qty: float):
        """
        Place a market order.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            side (str): "Buy" or "Sell"
            qty (float): Order quantity
        """
        try:
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side.capitalize(),
                orderType="Market",
                qty=str(qty)
            )
            if response.get("retCode") != 0:
                logger.error(f"API error placing order for {symbol}: {response.get('retMsg')}")
                raise Exception(f"Order failed: {response.get('retMsg')}")
            logger.info(f"Order placed successfully: {response}")
            return response
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            raise

    def get_order_book(self, symbol: str) -> dict | None:
        """
        Fetch the order book for a given symbol from Bybit.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")

        Returns:
            dict | None: Order book data with 'bids' and 'asks' if successful, None otherwise
        """
        try:
            response = self.client.get_orderbook(category="linear", symbol=symbol, limit=5)
            if response.get("retCode") != 0:
                logger.error(f"API error fetching order book for {symbol}: {response.get('retMsg')}")
                return None
            result = response['result']
            order_book = {
                'bids': [(str(bid[0]), str(bid[1])) for bid in result['b']],
                'asks': [(str(ask[0]), str(ask[1])) for ask in result['a']]
            }
            logger.info(f"Fetched order book for {symbol}")
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, interval: str = "60", limit: int = 200) -> list:
        """
        Fetch historical candlestick data for a symbol.

        Args:
            symbol (str): Trading pair (e.g., "BTCUSDT")
            interval (str): Candlestick interval (e.g., "1", "5", "15", "60")
            limit (int): Number of candles to fetch (max 200)

        Returns:
            list: List of OHLCV data if successful, empty list otherwise
        """
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            if response.get("retCode") != 0:
                logger.error(f"API error fetching historical data for {symbol}: {response.get('retMsg')}")
                return []
            data = response["result"]["list"]
            logger.info(f"Fetched {len(data)} historical data points for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

def main():
    api_key = "05EqRWk80CvjiSto64"
    api_secret = "6OhCdDGX7JQGePrqWd5Axl2q7k5SPNccprtH"
    bybit_api = BybitAPI(api_key, api_secret, testnet=True)

    price = bybit_api.get_btc_price()
    if price:
        logger.info(f"BTC price: {price}")

    trades = bybit_api.get_recent_trades("BTCUSDT")
    logger.info(f"Recent Trades: {trades}")

    order_book = bybit_api.get_order_book("BTCUSDT")
    if order_book:
        logger.info(f"Order Book: {order_book}")

    historical_data = bybit_api.get_historical_data("BTCUSDT")
    logger.info(f"Historical Data (first 5): {historical_data[:5]}")

    time.sleep(10)

if __name__ == "__main__":
    main()
