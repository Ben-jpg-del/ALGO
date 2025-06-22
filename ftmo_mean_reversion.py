import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from collections import deque
import logging

import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables from .env file if needed (e.g., for login parameters)
load_dotenv()

# Initialize MetaTrader5 connection
if not mt5.initialize():
    logging.error(f"initialize() failed, error code: {mt5.last_error()}")
    quit()

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
        # Note: In MT5, symbol names may differ. For example, "USD_CHF" might need to be "USDCHF".
        self.fx_pairs = ["USDCHF"]  # Use the exact symbol names as they appear in your terminal
        self.symbols = []  # Will store instruments that are actually available

        # Risk Management Parameters
        self.account_balance = 0  
        self.margin_percentage = 0.15  # 15% of the account balance
        self.leverage = 30             # 1:30 leverage

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
        Fetch the current account balance from MT5 and update the position size.
        """
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info from MT5.")
        else:
            balance = account_info.balance
            self.account_balance = balance
            # Calculate notional position size in USD:
            # For example, with balance ≈9720.95, margin=0.15, and leverage=30, 
            # position size = 9720.95 * 0.15 * 30 ≈ 43744.28 units.
            self.position_size_usd = self.account_balance * self.margin_percentage * self.leverage
            self.positionSize = self.position_size_usd
            logging.info(f"Updated account balance: {self.account_balance:.2f}, "
                         f"new position size: {self.positionSize:.2f}")

    def list_available_instruments(self):
        """
        Check if each desired instrument is available in MT5 and select it.
        """
        for fx in self.fx_pairs:
            symbol_info = mt5.symbol_info(fx)
            if symbol_info is None:
                logging.warning(f"Instrument {fx} not available in the account.")
            else:
                self.symbols.append(fx)
                self.SetLeverage(fx, self.leverage)
                mt5.symbol_select(fx, True)
                logging.info(f"Selected symbol: {fx}")
        # Initialize price history for each symbol
        self.priceHistory = {s: deque(maxlen=self.windowSize) for s in self.symbols}

    def SetLeverage(self, symbol, leverage):
        """
        Placeholder. In MT5 leverage settings are typically set at the account level.
        """
        pass

    def get_historical_data(self, symbol, granularity='M1', count=200):
        """
        Retrieve historical rates from MT5 for the specified instrument.
        """
        # Use MT5's one-minute timeframe (adjust if needed)
        timeframe = mt5.TIMEFRAME_M1
        # Get the most recent 'count' bars
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logging.error(f"Error fetching historical data for {symbol}")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        # Convert time (in seconds) to datetime in UTC
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        # Rename columns to match your OANDA code logic
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low',
                           'close': 'close', 'tick_volume': 'volume'}, inplace=True)
        return df

    def calculate_order_quantity(self, symbol, direction):
        """
        Return positionSize (in lots) with the correct sign:
        + for buy, - for sell.
        Assumes 100,000 units per lot.
        Adjusts the volume to meet the symbol's volume_min and volume_step.
        """
        # Compute raw lot size based on the notional position size (always positive)
        raw_lots = abs(self.positionSize) / 100000.0
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            min_volume = symbol_info.volume_min
            volume_step = symbol_info.volume_step
            # Round the absolute lot size to the nearest valid step
            valid_lots_abs = max(min_volume, round(raw_lots / volume_step) * volume_step)
        else:
            valid_lots_abs = raw_lots  # Fallback if symbol info is unavailable
        # Reapply the trade direction: positive for buy, negative for sell.
        return valid_lots_abs if direction > 0 else -valid_lots_abs

    def get_current_position(self, symbol):
        """
        Returns net volume for the given symbol.
        In MT5, positions_get() returns a list of positions.
        """
        positions = mt5.positions_get(symbol=symbol)
        pos_total = 0
        if positions:
            for pos in positions:
                # In MT5, pos.type == 0 means BUY, pos.type == 1 means SELL.
                if pos.symbol == symbol:
                    if pos.type == 0:
                        pos_total += pos.volume
                    else:
                        pos_total -= pos.volume
        logging.info(f"For {symbol}: net position = {pos_total}")
        return pos_total

    def get_all_open_trades(self, symbol):
        """
        Retrieve all open positions for a given symbol.
        """
        positions = mt5.positions_get(symbol=symbol)
        trades = []
        if positions:
            for pos in positions:
                trades.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'volume': pos.volume,
                    'type': pos.type  # 0 for BUY, 1 for SELL
                })
        logging.info(f"Open trades for {symbol}: {trades}")
        return trades

    def close_trade(self, trade_id):
        """
        Close a specific trade by trade ticket.
        """
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if pos.ticket == trade_id:
                    symbol = pos.symbol
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        logging.error(f"Failed to get tick for {symbol}")
                        return
                    # For a BUY position, closing order is a SELL; for a SELL, a BUY.
                    if pos.type == 0:  # BUY position
                        price = tick.bid
                        order_type = mt5.ORDER_TYPE_SELL
                    else:  # SELL position
                        price = tick.ask
                        order_type = mt5.ORDER_TYPE_BUY
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": order_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "python script close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    logging.info(f"Closing trade for {symbol}: Request parameters: {request}")
                    result = mt5.order_send(request)
                    logging.info(f"Close trade response for {trade_id}: {result}")
                    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                        logging.error(f"Error closing trade {trade_id}: {result.comment if result else 'No result returned'}")
                    else:
                        logging.info(f"Successfully closed trade ID: {trade_id}")
                    break

    def place_order(self, symbol, units, side='buy'):
        """
        Submit a MARKET order in MT5.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"Failed to get tick for {symbol}")
            return False

        # For a BUY order, use the ask price; for SELL, use the bid price.
        price = tick.ask if side == 'buy' else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": abs(units),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        logging.info(f"Placing order for {symbol}: Request parameters: {request}")
        result = mt5.order_send(request)
        logging.info(f"Order response for {symbol}: {result}")
        if result is None:
            logging.error(f"Order send returned None for {symbol}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Error placing order for {symbol}: {result.comment}")
            return False
        else:
            logging.info(f"Order placed - {side.upper()} {abs(units)} lots of {symbol}")
            return True

    def get_current_bid_ask(self, symbol):
        """
        Retrieve current bid and ask prices for the symbol.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"Error retrieving current pricing for {symbol}")
            return None, None
        logging.info(f"Pricing for {symbol}: bid = {tick.bid}, ask = {tick.ask}")
        return tick.bid, tick.ask

    def get_current_price(self, symbol, side=None):
        """
        Returns the current market price for the symbol.
        If side is specified ('long' or 'short'), returns the appropriate price.
        Otherwise, falls back to the last candle's close.
        """
        bid, ask = self.get_current_bid_ask(symbol)
        if bid is None or ask is None:
            if len(self.priceHistory[symbol]) == 0:
                return 0
            return self.priceHistory[symbol][-1]
        
        if side == 'long':
            # For a long position, use the ask price.
            return ask
        elif side == 'short':
            # For a short position, use the bid price.
            return bid
        else:
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

            # Retrieve all open trades (positions) for this symbol:
            open_trades = self.get_all_open_trades(symbol)

            entry_time = self.entryTimeBySymbol.get(symbol, None)
            if entry_time is None:
                continue

            holding_duration = (datetime.now(timezone.utc) - entry_time).total_seconds()
            if holding_duration > self.maxHoldingDays * 24 * 3600:
                # Time-based exit -> Close all open trades for the symbol
                for trade in open_trades:
                    self.close_trade(trade_id=trade['ticket'])
                self.entryTimeBySymbol.pop(symbol, None)
                logging.info(f"{symbol} => Time-based exit after {self.maxHoldingDays} days.")
                continue

            try:
                rolling_mean = np.mean(self.priceHistory[symbol])
                side = 'long' if current_position > 0 else 'short'
                current_price = self.get_current_price(symbol, side=side)

                if current_position > 0 and current_price >= rolling_mean:
                    for trade in open_trades:
                        self.close_trade(trade_id=trade['ticket'])
                    self.entryTimeBySymbol.pop(symbol, None)
                    logging.info(f"{symbol} => Price reached the mean ({rolling_mean:.4f}), closed LONG.")

                elif current_position < 0 and current_price <= rolling_mean:
                    for trade in open_trades:
                        self.close_trade(trade_id=trade['ticket'])
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
            self.update_account_balance()

            for symbol in self.symbols:
                df = self.get_historical_data(symbol, granularity='M1', count=self.windowSize)
                if not df.empty:
                    self.priceHistory[symbol].extend(df['close'].tolist())

            if all(len(self.priceHistory[s]) >= self.windowSize for s in self.symbols):
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
    try:
        strategy.run()
    except KeyboardInterrupt:
        logging.info("Strategy interrupted by user.")
    finally:
        mt5.shutdown()
