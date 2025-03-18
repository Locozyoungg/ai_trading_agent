# bybit_client.py
"""
The Pinnacle of Trading AI: Bybit Client with Global Resilience and Self-Healing
"""

import ccxt
import os
import time
import logging
import json
import websocket
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.sessions import Session
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AdvancedConnectionManager:
    """Manages connections with domain-based failover."""
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.endpoints = {
            'rest': [
                'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com',
                'https://api-testnet.bybit.com' if testnet else 'https://api.bybit.com'
            ],
            'ws': [
                'wss://stream-testnet.bybit.com/v5/public/linear' if testnet else 'wss://stream.bybit.com/v5/public/linear',
                'wss://stream-testnet.bybit.com/v5/public/linear' if testnet else 'wss://stream.bybit.com/v5/public/linear'
            ]
        }
        self.active_rest = 0
        self.active_ws = 0
        self.lock = threading.Lock()

    def get_rest_endpoint(self) -> str:
        with self.lock:
            return self.endpoints['rest'][self.active_rest]

    def get_ws_endpoint(self) -> str:
        with self.lock:
            return self.endpoints['ws'][self.active_ws]

    def switch_rest_endpoint(self):
        with self.lock:
            self.active_rest = (self.active_rest + 1) % len(self.endpoints['rest'])
            logger.info(f"Switched REST endpoint to {self.endpoints['rest'][self.active_rest]}")

    def switch_ws_endpoint(self):
        with self.lock:
            self.active_ws = (self.active_ws + 1) % len(self.endpoints['ws'])
            logger.info(f"Switched WS endpoint to {self.endpoints['ws'][self.active_ws]}")

class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        logger.debug(f"Initializing BybitClient with api_key={api_key[:4]}..., testnet={testnet}")
        self.conn_manager = AdvancedConnectionManager(testnet)
        self.testnet = testnet
        self.client = self._init_ccxt_client(api_key, api_secret)
        self.ws = None
        self.ws_connected = False
        self.ws_thread = None
        self.order_book_buffer = deque(maxlen=50)
        self.trade_buffer = deque(maxlen=200)
        self.order_book = {'bids': [], 'asks': [], 'timestamp': 0}
        self.last_update_time = 0
        self.lock = threading.Lock()
        self.ws_retries = 0
        self.max_ws_retries = 10
        self.ws_retry_delay = 5
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("BybitClient initialized with domain-based resilience")

    def _init_ccxt_client(self, api_key: str, api_secret: str) -> ccxt.bybit:
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'test': self.testnet,
            'options': {'defaultType': 'linear', 'adjustForTimeDifference': True},
            'verbose': True
        }
        session = Session()
        retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=50)
        session.mount('https://', adapter)
        session.headers.update({'Connection': 'keep-alive'})
        client = ccxt.bybit(config)
        client.session = session
        client.milliseconds = lambda: int(time.time() * 1000)
        base_url = self.conn_manager.get_rest_endpoint()
        client.urls['api'] = {'public': base_url, 'private': base_url}
        return client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_websocket()
        self.client.session.close()
        self.executor.shutdown(wait=True)
        if exc_type:
            logger.error(f"Context exit with error: {exc_type}: {exc_value}", exc_info=True)

    def fetch_with_retry(self, method: str, *args, max_retries: int = 5, delay: int = 3, **kwargs) -> Optional[dict]:
        logger.debug(f"Fetching with retry: method={method}, args={args}, kwargs={kwargs}")
        for attempt in range(1, max_retries + 1):
            try:
                base_url = self.conn_manager.get_rest_endpoint()
                self.client.urls['api']['public'] = base_url
                self.client.urls['api']['private'] = base_url
                response = getattr(self.client, method)(*args, **kwargs)
                if isinstance(response, dict) and 'retCode' in response and response['retCode'] != 0:
                    raise Exception(f"API error: {response['retMsg']}")
                logger.info(f"Fetch {method} succeeded")
                return response
            except Exception as e:
                logger.error(f"Attempt {attempt}/{max_retries} failed: {str(e)}", exc_info=True)
                if attempt == max_retries:
                    return None
                self.conn_manager.switch_rest_endpoint()
                time.sleep(delay * (2 ** (attempt - 1)))
        return None

    def get_balance(self) -> float:
        balance = self.fetch_with_retry('fetch_balance', {'type': 'future'})
        return float(balance['total'].get('USDT', 0.0)) if balance else 0.0

    def place_order(self, symbol: str, qty: float, side: str, order_type: str = "Market", price: Optional[float] = None, reduce_only: bool = False) -> Dict:
        params = {'reduceOnly': reduce_only, 'category': 'linear'}
        if order_type.lower() == "limit" and price is not None:
            params['price'] = price
        logger.debug(f"Placing order: symbol={symbol}, qty={qty}, side={side}, order_type={order_type}, reduce_only={reduce_only}")
        order = self.fetch_with_retry('create_order', symbol, order_type.lower(), side.lower(), qty, params=params)
        if order and 'id' in order:
            logger.info(f"Order placed: {side} {qty} {symbol} (reduce_only={reduce_only})")
        else:
            logger.error(f"Order placement failed: {order}")
        return order or {}

    def get_current_price(self, symbol: str) -> float:
        ticker = self.fetch_with_retry('fetch_ticker', symbol)
        return float(ticker['last']) if ticker else 0.0

    def get_historical_data(self, symbol: str, interval: str = "1", limit: int = 50) -> np.ndarray:
        ohlcv = self.fetch_with_retry('fetch_ohlcv', symbol, timeframe=interval, limit=limit + 5, params={'category': 'linear'})
        if ohlcv and len(ohlcv) >= limit:
            return np.array([[float(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])] for item in ohlcv[-limit:]])
        logger.warning(f"Insufficient OHLCV data for {symbol}. Returning zeros.")
        return np.zeros((limit, 6))

    def close_position(self, symbol: str) -> Optional[Dict]:
        positions = self.get_positions(symbol)
        if not positions:
            logger.info(f"No open positions found for {symbol}")
            return None
        position = next((p for p in positions if float(p.get('contracts', 0)) > 0), None)
        if not position:
            logger.info(f"No active position to close for {symbol}")
            return None
        current_side = position['side'].lower()
        reduce_side = 'sell' if current_side == 'long' else 'buy'
        qty = float(position['contracts'])
        if qty <= 0:
            logger.warning(f"Invalid position size ({qty}) for {symbol}. Skipping close.")
            return None
        logger.debug(f"Closing {current_side} position of {qty} {symbol} with {reduce_side} order")
        order = self.place_order(symbol=symbol, qty=qty, side=reduce_side, order_type="Market", reduce_only=True)
        if order and 'id' in order:
            logger.info(f"Successfully closed position: {position}")
            return order
        else:
            logger.error(f"Failed to close position: {position}. Order response: {order}")
            return None

    def get_positions(self, symbol: str) -> List[Dict]:
        positions = self.fetch_with_retry('fetch_positions', [symbol], params={'category': 'linear'})
        if positions:
            return [{
                'symbol': pos.get('symbol', symbol),
                'contracts': str(pos.get('contracts', '0')),
                'side': pos.get('side', 'None'),
                'entryPrice': str(pos.get('entryPrice', '0')),
                'leverage': str(pos.get('leverage', '0')),
                'timestamp': pos.get('timestamp', self.client.milliseconds()),
                'unrealisedPnl': str(pos.get('unrealisedPnl', '0'))
            } for pos in positions if pos.get('contracts') is not None]
        logger.debug(f"No positions found for {symbol}")
        return []

    def get_order_book(self, symbol: str) -> Dict[str, List[List[float]]]:
        order_book = self.fetch_with_retry('fetch_order_book', symbol, limit=25, params={'category': 'linear'})
        if order_book and 'bids' in order_book and 'asks' in order_book:
            processed = {
                'bids': order_book['bids'][:25],
                'asks': order_book['asks'][:25],
                'timestamp': self.client.milliseconds()
            }
            with self.lock:
                self.order_book_buffer.append(processed)
                self.order_book = processed
                self.last_update_time = processed['timestamp']
            return processed
        logger.warning(f"Failed to fetch order book for {symbol}")
        return {'bids': [], 'asks': [], 'timestamp': self.client.milliseconds()}

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        trades = self.fetch_with_retry('fetch_trades', symbol, limit=limit, params={'category': 'linear'})
        if trades:
            with self.lock:
                self.trade_buffer.extend(trades)
            return trades
        logger.warning(f"Failed to fetch trades for {symbol}")
        return []

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "success" in data:
                logger.info(f"Subscription {data.get('op')} {'succeeded' if data['success'] else 'failed'}")
                return
            topic = data.get("topic", "")
            timestamp = data.get("ts", self.client.milliseconds())
            if "orderbook" in topic:
                order_data = data.get("data", {})
                bids = order_data.get("b", [])[:25] or []
                asks = order_data.get("a", [])[:25] or []
                processed = {
                    'bids': bids + [[0, 0]] * (25 - len(bids)) if len(bids) < 25 else bids,
                    'asks': asks + [[0, 0]] * (25 - len(asks)) if len(asks) < 25 else asks,
                    'timestamp': timestamp
                }
                with self.lock:
                    self.order_book_buffer.append(processed)
                    self.order_book = processed
                    self.last_update_time = timestamp
            elif "trade" in topic:
                trade_data = data.get("data", [])
                if trade_data:
                    with self.lock:
                        self.trade_buffer.extend(trade_data)
        except Exception as e:
            logger.error(f"Message processing failed: {str(e)}", exc_info=True)

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}", exc_info=True)
        with self.lock:
            self.ws_connected = False
        self._retry_websocket()

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code}, {close_msg}")
        with self.lock:
            self.ws_connected = False
        self._retry_websocket()

    def on_open(self, ws):
        with self.lock:
            self.ws_connected = True
            self.ws_retries = 0
        logger.info("WebSocket connection opened")
        ws.send(json.dumps({"op": "subscribe", "args": ["orderbook.25.BTCUSDT", "trade.BTCUSDT"]}))

    def _retry_websocket(self):
        with self.lock:
            if self.ws_retries >= self.max_ws_retries:
                logger.error(f"Max WebSocket retries ({self.max_ws_retries}) reached. Giving up.")
                return
            self.ws_retries += 1
            logger.info(f"Retrying WebSocket {self.ws_retries}/{self.max_ws_retries}")
            time.sleep(self.ws_retry_delay * (2 ** (self.ws_retries - 1)))
            self.conn_manager.switch_ws_endpoint()
            self.start_websocket()

    def start_websocket(self):
        with self.lock:
            if self.ws_connected:
                logger.warning("WebSocket already running")
                return
            if self.ws and self.ws_thread and self.ws_thread.is_alive():
                logger.info("Closing existing WebSocket connection before starting a new one")
                self.stop_websocket()
                time.sleep(2)
        ws_url = self.conn_manager.get_ws_endpoint()
        logger.debug(f"Starting WebSocket: {ws_url}")
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
            header={'Host': 'stream-testnet.bybit.com' if self.testnet else 'stream.bybit.com'}
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
        logger.info("WebSocket thread started")

    def stop_websocket(self):
        with self.lock:
            if self.ws and self.ws_connected:
                self.ws.close()
                self.ws_connected = False
                logger.info("WebSocket stopped")
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
                if self.ws_thread.is_alive():
                    logger.warning("WebSocket thread did not terminate gracefully")
            self.ws = None
            self.ws_thread = None

    def get_latest_order_book(self) -> Dict[str, List[List[float]]]:
        with self.lock:
            if self.last_update_time > self.client.milliseconds() - 1000 and self.order_book['bids']:
                return self.order_book
        return self.get_order_book("BTCUSDT")

    def get_latest_trades(self) -> List[Dict]:
        with self.lock:
            return list(self.trade_buffer)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = BybitClient(os.getenv('BYBIT_API_KEY'), os.getenv('BYBIT_API_SECRET'), testnet=True)
    client.start_websocket()
    time.sleep(10)
    print(client.get_latest_order_book())
    print(client.get_latest_trades()[:5])
    client.stop_websocket()
