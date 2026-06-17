# strategies/sell_strategy.py
from strategies.trading_strategy import AdvancedTradingStrategy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SellStrategy(AdvancedTradingStrategy):
    def __init__(self, client=None, symbol=None):
        super().__init__(client=client, symbol=symbol)
        logger.info(f"SellStrategy initialized for {self.symbol}")

    def execute_trade(self, qty=None):
        qty = qty or 0.001  # Default to 0.001 if not provided
        logger.info(f"Executing Sell trade for {self.symbol} with qty={qty}")
        if self.client:
            self.client.place_order(self.symbol, qty=qty, side="Sell", order_type="Market")
