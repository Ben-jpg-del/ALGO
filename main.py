#region imports
from AlgorithmImports import *

from itertools import combinations
import statsmodels.tsa.stattools as ts

from pair import Pairs
from symbol_data import SymbolData
from trading_pair import TradingPair
#endregion


class PairsTrading(QCAlgorithm):     

    _symbols = [
            'ING', 'TBC', 'BMA', 'PB', 'FBC', 'STL', 'FCF', 'PFS', 'BOH', 'SCNB',
            'BK', 'CMA', 'AF', 'PNC', 'KB', 'SHG', 'BSAC', 'CIB', 'BBD', 'BSBR'
        ]
    _num_bar = 390*21*3
    _interval = 10
    _pair_num = 10
    _leverage = 1
    _min_corr_threshold = 0.9
    _open_size = 2.32
    _close_size = 0.5
    _stop_loss_size = 6

    def initialize(self):
        self.set_start_date(2024, 9, 1)
        self.set_end_date(2024, 10, 1)
        self.set_cash(50000)
        
        self._symbol_data = {}
        self._pair_list = []
        self._selected_pair = []
        self._trading_pairs = {}
        self._regenerate_time = datetime.min

        for ticker in self._symbols:
            symbol = self.add_equity(ticker, Resolution.MINUTE).symbol
            self._symbol_data[symbol] = SymbolData(self, symbol, self._num_bar, self._interval)
        
        for pair in combinations(self._symbol_data.items(), 2):
            if pair[0][1].is_ready and pair[1][1].is_ready:
                self._pair_list.append(Pairs(pair[0][1], pair[1][1]))

    def _generate_pairs(self):
        selected_pair = []
        for pair in self._pair_list:
            # correlation selection 
            if pair.correlation() < self._min_corr_threshold:
                continue

            # cointegration selection 
            coint = pair.cointegration_test()
            if coint and pair.stationary_p < 0.05:
                selected_pair.append(pair)

        if len(selected_pair) == 0:
            self.debug('No selected pair')
            return []

        selected_pair.sort(key = lambda x: x.correlation(), reverse = True)
        if len(selected_pair) > self._pair_num:
            selected_pair = selected_pair[:self._pair_num]
        selected_pair.sort(key = lambda x: x.stationary_p)

        return selected_pair

    def on_data(self, data):
        for symbol, symbolData in self._symbol_data.items():
            if data.bars.contains_key(symbol):
                symbolData.update(data.bars[symbol])

        # generate pairs with correlation and cointegration selection 
        if self._regenerate_time < self.time:
            self._selected_pair = self._generate_pairs()
            self._regenerate_time = self.time + timedelta(days=5)

        # closing existing position
        for pair, trading_pair in self._trading_pairs.copy().items():
            # close: if not correlated nor cointegrated anymore
            if pair not in self._selected_pair:
                self.market_order(pair.a.symbol, -trading_pair.ticket_a.quantity)
                self.market_order(pair.b.symbol, -trading_pair.ticket_b.quantity)
                self._trading_pairs.pop(pair)
                self.debug(f'Close {pair.name}')
                continue

            # get current cointegrated series deviation from mean
            error = pair.a.prices[0].close - (trading_pair.model_intercept + trading_pair.model_slope * pair.b.prices[0].close)
            
            # close: when the cointegrated series is deviated less than 0.5 SD from its mean
            if (trading_pair.ticket_a.quantity > 0 and
                (error > trading_pair.mean_error - self._close_size * trading_pair.epsilon or 
                error < trading_pair.mean_error - self._stop_loss_size * trading_pair.epsilon)):
                self.market_order(pair.a.symbol, -trading_pair.ticket_a.quantity)
                self.market_order(pair.b.symbol, -trading_pair.ticket_b.quantity)
                self._trading_pairs.pop(pair)
                self.debug(f'Close {pair.name}')

            elif (trading_pair.ticket_a.quantity < 0 and 
                (error < trading_pair.mean_error + self._close_size * trading_pair.epsilon or 
                error > trading_pair.mean_error + self._stop_loss_size * trading_pair.epsilon)):
                self.market_order(pair.a.symbol, -trading_pair.ticket_a.quantity)
                self.market_order(pair.b.symbol, -trading_pair.ticket_b.quantity)
                self._trading_pairs.pop(pair)
                self.debug(f'Close {pair.name}')

        # entry: when the cointegrated series is deviated by more than 2.32 SD from its mean
        for pair in self._selected_pair:
            # get current cointegrated series deviation from mean
            price_a = pair.a.prices[0].close
            price_b = pair.b.prices[0].close
            error = price_a - (pair.model.params[0] + pair.model.params[1] * price_b)

            if pair not in self._trading_pairs:
                if error < pair.mean_error - self._open_size * pair.epsilon:
                    qty_a = self.calculate_order_quantity(symbol, self._leverage/self._pair_num / 2)
                    qty_b = self.calculate_order_quantity(symbol, -self._leverage/self._pair_num / 2)
                    ticket_a = self.market_order(pair.a.symbol, qty_a)
                    ticket_b = self.market_order(pair.b.symbol, qty_b)
                    
                    self._trading_pairs[pair] = TradingPair(ticket_a, ticket_b, pair.model.params[0], pair.model.params[1], pair.mean_error, pair.epsilon)
                    self.debug(f'Long {qty_a} {pair.a.symbol.value} and short {qty_b} {pair.b.symbol.value}')

                elif error > pair.mean_error + self._open_size * pair.epsilon:
                    qty_a = self.calculate_order_quantity(symbol, -self._leverage/self._pair_num / 2)
                    qty_b = self.calculate_order_quantity(symbol, self._leverage/self._pair_num / 2)
                    ticket_a = self.market_order(pair.a.symbol, qty_a)
                    ticket_b = self.market_order(pair.b.symbol, qty_b)
                    
                    self._trading_pairs[pair] = TradingPair(ticket_a, ticket_b, pair.model.params[0], pair.model.params[1], pair.mean_error, pair.epsilon)
                    self.debug(f'Long {qty_b} {pair.b.symbol.value} and short {qty_a} {pair.a.symbol.value}')
        
