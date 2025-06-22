#region imports
from AlgorithmImports import *

import statsmodels.formula.api as sm
from statsmodels.tsa.stattools import coint, adfuller
#endregion


class Pairs(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = f'{a.symbol.value}:{b.symbol.value}'

        self.model = None
        self.mean_error = 0
        self.epsilon = 0

    def _data_frame(self):
        df = pd.concat([self.a.data_frame.droplevel([0]), self.b.data_frame.droplevel([0])], axis=1).dropna()
        df.columns = [self.a.symbol.value, self.b.symbol.value]
        return df
    
    def correlation(self):
        return self._data_frame().corr().iloc[0][1]

    def cointegration_test(self):
        coint_test = coint(self.a.series.values.flatten(), self.b.series.values.flatten(), trend="n", maxlag=0)

        # Return if not cointegrated
        if coint_test[1] >= 0.05:
            return False

        self.model = sm.ols(formula = f'{self.a.symbol.value} ~ {self.b.symbol.value}', data=self._data_frame()).fit()
        self.stationary_p = adfuller(self.model.resid, autolag = 'BIC')[1]
        self.mean_error = np.mean(self.model.resid)
        self.epsilon = np.std(self.model.resid)
        
        return True
        