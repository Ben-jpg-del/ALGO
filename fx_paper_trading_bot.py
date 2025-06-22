import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from collections import deque
import logging

from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountInstruments, AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.positions import PositionList
from oandapyV20.endpoints.trades import TradesList, TradeClose
from oandapyV20.endpoints.pricing import PricingInfo
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OANDA_API_TOKEN = os.getenv('OANDA_API_TOKEN')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
OANDA_API_URL = os.getenv('OANDA_API_URL')  # e.g. 'https://api-fxpractice.oanda.com'

# Initialize OANDA API in live mode
api = API(access_token=OANDA_API_TOKEN, environment="practice")

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("fx_trading_bot.log"),
        logging.StreamHandler()
    ]
)

class MovingMeanReversion:
    def __init__(self):
        # Strategy Parameters
        self.fx_pairs = ["USD_CHF", "USD_NOK"]  # OANDA instrument names
        self.symbols = []  # Will store instruments that are actually available

        # Risk Management Parameters
        # The account balance will be updated via the API.
        self.account_balance = 0  
        self.margin_percentage = 0.04       # 6% margin
        self.leverage = 50                  # 50:1 leverage

        # Position size variables (will be updated after fetching the balance)
        self.position_size_usd = 0  
        self.positionSize = 0

        self.windowDays = 2
        self.windowSize = self.windowDays * 1440  # 2 days * 1440 minutes/day

        self.deviationThreshold = 0.005
        self.maxHoldingDays = 0.5  # hold up to 0.5 days
        self.entryTimeBySymbol = {}

        # Trade frequency rules
        self.maxTradesPerDay = 2
        self.tradesTodayCount = 0
        self.currentDay = None

        logging.info("Initialized simple moving-mean reversion strategy.")
        self.list_available_instruments()

        # Update account balance (and recalc position sizes) at initialization.
        self.update_account_balance()

    def update_account_balance(self):
        """
        Fetch the current account balance from the OANDA API and update the position size.
        """
        try:
            details_endpoint = AccountDetails(accountID=OANDA_ACCOUNT_ID)
            response = api.request(details_endpoint)
            # The balance is returned as a string, so convert it to float.
            balance = float(response['account']['balance'])
            self.account_balance = balance
            # Recalculate the notional position size based on current balance.
            self.position_size_usd = self.account_balance * self.margin_percentage * self.leverage
            self.positionSize = self.position_size_usd
            logging.info(f"Updated account balance: {self.account_balance:.2f}, "
                         f"new position size: {self.positionSize:.2f}")
        except Exception as e:
            logging.error(f"Error updating account balance: {e}")

    def list_available_instruments(self):
        try:
            instruments_endpoint = AccountInstruments(accountID=OANDA_ACCOUNT_ID)
            response = api.request(instruments_endpoint)
            available_instruments = [instr['name'] for instr in response.get('instruments', [])]
            logging.info("Available Instruments:")
            for instr in available_instruments:
                logging.info(instr)

            # Pick only those from fx_pairs that are truly available
            for fx in self.fx_pairs:
                if fx in available_instruments:
                    self.symbols.append(fx)
                    self.SetLeverage(fx, self.leverage)
                else:
                    logging.warning(f"Instrument {fx} not available in the account.")

            # Initialize price history for each symbol
            self.priceHistory = {s: deque(maxlen=self.windowSize) for s in self.symbols}

        except Exception as e:
            logging.error(f"Error fetching available instruments: {e}")

    def SetLeverage(self, symbol, leverage):
        """
        Placeholder. OANDA does not allow adjusting leverage via API.
        """
        pass

    def get_historical_data(self, symbol, granularity='M1', count=200):
        """
        Retrieve candles from OANDA for the specified instrument.
        """
        params = {
            "granularity": granularity,
            "count": count
        }
        r = InstrumentsCandles(instrument=symbol, params=params)
        try:
            response = api.request(r)
            data = response['candles']
            df = pd.DataFrame([{
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': candle['volume']
            } for candle in data])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_order_quantity(self, symbol, direction):
        """
        Return positionSize with the correct sign:
        + for buy, - for sell.
        """
        return int(direction * self.positionSize)

    def get_current_position(self, symbol):
        """
        Returns net units (long plus short, since short units are negative) for the given symbol.
        Logs the raw API response for debugging.
        """
        pos = PositionList(accountID=OANDA_ACCOUNT_ID)
        try:
            response = api.request(pos)
            logging.info(f"Positions API response for {symbol}: {response}")
            for position in response.get('positions', []):
                if position['instrument'] == symbol:
                    long_units = float(position['long']['units'])
                    short_units = float(position['short']['units'])
                    logging.info(f"For {symbol}: long_units = {long_units}, short_units = {short_units}")
                    return long_units + short_units
            return 0
        except Exception as e:
            logging.error(f"Error fetching position for {symbol}: {e}")
            return 0

    def get_all_open_trades(self, symbol):
        """
        Retrieve all open trades for a given symbol.
        """
        trades_endpoint = TradesList(accountID=OANDA_ACCOUNT_ID, params={"instrument": symbol})
        try:
            response = api.request(trades_endpoint)
            return response.get("trades", [])
        except Exception as e:
            logging.error(f"Error fetching trade list for {symbol}: {e}")
            return []

    def close_trade(self, trade_id):
        """
        Close a specific trade by trade ID.
        """
        close_endpoint = TradeClose(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
        try:
            api.request(close_endpoint)
            logging.info(f"Successfully closed trade ID: {trade_id}")
        except Exception as e:
            logging.error(f"Error closing trade ID {trade_id}: {e}")

    def place_order(self, symbol, units, side='buy'):
        """
        Submit a MARKET order with no OANDA-based SL/TP.
        """
        order_data = {
            "order": {
                "units": str(abs(units)),
                "instrument": symbol,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        if side == 'sell':
            order_data["order"]["units"] = "-" + order_data["order"]["units"]

        r = OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
        try:
            response = api.request(r)
            if response.get('orderFillTransaction', {}).get('id'):
                logging.info(f"Order placed - {side.upper()} {abs(units)} units of {symbol}")
                return True
            else:
                logging.warning(f"Order not filled for {symbol}")
                return False
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}")
            return False

    def get_current_bid_ask(self, symbol):
        """
        Retrieve current bid and ask prices for the symbol using OANDA's pricing API.
        """
        params = {"instruments": symbol}
        r = PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        try:
            response = api.request(r)
            logging.info(f"Pricing API response for {symbol}: {response}")
            prices = response.get('prices', [])
            if prices:
                bid = float(prices[0]['bids'][0]['price'])
                ask = float(prices[0]['asks'][0]['price'])
                return bid, ask
            else:
                return None, None
        except Exception as e:
            logging.error(f"Error retrieving current pricing for {symbol}: {e}")
            return None, None

    def get_current_price(self, symbol, side=None):
        """
        Returns the current market price for the symbol.
        If side is specified ('long' or 'short'), returns the appropriate price.
        Otherwise, falls back to the last candle's close.
        """
        bid, ask = self.get_current_bid_ask(symbol)
        if bid is None or ask is None:
            # Fallback to the stored candle data
            if len(self.priceHistory[symbol]) == 0:
                return 0
            return self.priceHistory[symbol][-1]
        
        if side == 'long':
            # For a long position, use the ask price (price to buy).
            return ask
        elif side == 'short':
            # For a short position, use the bid price (price to sell).
            return bid
        else:
            # Default: return the average of bid and ask
            return (bid + ask) / 2

    def manage_open_positions(self):
        """
        Checks how long each open position has been held; exit if
        time-based rule or mean-reversion rule is met.
        """
        for symbol in self.symbols:
            current_position = self.get_current_position(symbol)
            if current_position == 0:
                continue

            # Retrieve all open trades for this symbol:
            open_trades = self.get_all_open_trades(symbol)

            entry_time = self.entryTimeBySymbol.get(symbol, None)
            if entry_time is None:
                continue

            holding_duration = (datetime.now(timezone.utc) - entry_time).total_seconds()
            # 1 day = 24 * 3600 seconds
            if holding_duration > self.maxHoldingDays * 24 * 3600:
                # Time-based exit -> Close all open trades for the symbol
                for trade in open_trades:
                    self.close_trade(trade_id=trade['id'])
                # Remove the symbol's entry time since we're fully exiting
                self.entryTimeBySymbol.pop(symbol, None)
                logging.info(f"{symbol} => Time-based exit after {self.maxHoldingDays} days.")
                continue

            try:
                rolling_mean = np.mean(self.priceHistory[symbol])
                # Determine the side of the position based on current_position sign.
                side = 'long' if current_position > 0 else 'short'
                current_price = self.get_current_price(symbol, side=side)

                # For LONG positions, exit when current price is at or above the mean.
                if current_position > 0 and current_price >= rolling_mean:
                    for trade in open_trades:
                        self.close_trade(trade_id=trade['id'])
                    self.entryTimeBySymbol.pop(symbol, None)
                    logging.info(f"{symbol} => Price reached the mean ({rolling_mean:.4f}), closed LONG.")

                # For SHORT positions, exit when current price is at or below the mean.
                elif current_position < 0 and current_price <= rolling_mean:
                    for trade in open_trades:
                        self.close_trade(trade_id=trade['id'])
                    self.entryTimeBySymbol.pop(symbol, None)
                    logging.info(f"{symbol} => Price reached the mean ({rolling_mean:.4f}), closed SHORT.")

            except Exception as e:
                logging.error(f"Error managing open positions for {symbol}: {e}")

    def execute_trades(self):
        """
        Checks if we can open a new position using the
        moving mean-reversion logic if we haven't reached
        maxTradesPerDay.
        """
        today = datetime.now(timezone.utc).date()
        if self.currentDay != today:
            self.currentDay = today
            self.tradesTodayCount = 0

        if self.tradesTodayCount >= self.maxTradesPerDay:
            return

        for symbol in self.symbols:
            if self.tradesTodayCount >= self.maxTradesPerDay:
                break

            current_position = self.get_current_position(symbol)
            if current_position != 0:
                continue  # Already in a position

            prices = list(self.priceHistory[symbol])
            if len(prices) < self.windowSize:
                continue  # Not enough historical data

            current_price = prices[-1]
            rolling_mean = np.mean(prices)
            if rolling_mean == 0:
                continue

            dev = (current_price - rolling_mean) / rolling_mean
            abs_dev = abs(dev)

            if abs_dev > self.deviationThreshold:
                # If price is above mean, we go short; if below mean, we go long.
                direction = -1 if dev > 0 else 1
                side = 'sell' if direction == -1 else 'buy'
                qty = self.calculate_order_quantity(symbol, direction)

                success = self.place_order(symbol, qty, side=side)
                if success:
                    self.tradesTodayCount += 1
                    self.entryTimeBySymbol[symbol] = datetime.now(timezone.utc)
                    logging.info(
                        f"{symbol} => dev={dev:.3%}, RollingMean={rolling_mean:.4f}, "
                        f"Entered {'LONG' if direction > 0 else 'SHORT'}, dailyTrades={self.tradesTodayCount}"
                    )

    def run(self):
        """
        Main loop:
        1) Update account balance
        2) Pull most recent price data for each symbol
        3) Manage existing open positions
        4) Possibly open new trades
        5) Sleep until next minute
        """
        while True:
            # Update the account balance (and order size) each loop iteration.
            self.update_account_balance()

            for symbol in self.symbols:
                df = self.get_historical_data(symbol, granularity='M1', count=self.windowSize)
                if not df.empty:
                    self.priceHistory[symbol].extend(df['close'].tolist())

            # Only proceed if all symbols have enough data
            if all(len(self.priceHistory[s]) >= self.windowSize for s in self.symbols):
                # Log the current deviation for each symbol
                for symbol in self.symbols:
                    rolling_mean = np.mean(self.priceHistory[symbol])
                    current_price = self.get_current_price(symbol)
                    if rolling_mean != 0:
                        dev = (current_price - rolling_mean) / rolling_mean
                        logging.info(f"{symbol} => Current Deviation: {dev:.5%}")

                self.manage_open_positions()
                self.execute_trades()

            time_to_sleep = 60 - datetime.now(timezone.utc).second
            time.sleep(time_to_sleep)

if __name__ == "__main__":
    strategy = MovingMeanReversion()
    strategy.run()
