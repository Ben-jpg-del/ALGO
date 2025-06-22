#region imports
from AlgorithmImports import *
#endregion


class TradingPair(object):

    def __init__(self, ticket_a, ticket_b, intercept, slope, mean_error, epsilon):
        self.ticket_a = ticket_a
        self.ticket_b = ticket_b

        self.model_intercept = intercept
        self.model_slope = slope

        self.mean_error = mean_error
        self.epsilon = epsilon