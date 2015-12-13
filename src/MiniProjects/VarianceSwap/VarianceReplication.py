import numpy as np

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class VarianceReplication(object):
    def __init__(self, tau: float, forward: float, b: float, strikes_put: np.array, prices_put: np.array,
                 strikes_call: np.array, prices_call: np.array):
        self.tau = tau
        self.forward = forward
        self.b = b
        self.strikes_put = strikes_put
        self.prices_put = prices_put
        self.strikes_call = strikes_call
        self.prices_call = prices_call
        pass

    def prepare_replication_portfolio(self):
        ks_p_otm, ps_p_otm = self.filter_otm_quotes(self.forward, self.strikes_put, self.prices_put, OptionType.put)
        ks_c_otm, ps_c_otm = self.filter_otm_quotes(self.forward, self.strikes_call, self.prices_call, OptionType.call)
        strikes = np.append(ks_p_otm, ks_c_otm)
        prices = np.append(ps_p_otm, ps_c_otm)
        return strikes, prices

    def calc_variance(self):
        # integral starts from zero till highest strike with piecewise constant evaluation towards left
        strikes, prices = self.prepare_replication_portfolio()
        increment = strikes[1:] - strikes[:-1]
        increment = np.insert(increment, 0, strikes[0])  # first element is accumulated from zero
        integral = np.sum(increment * prices / (strikes ** 2))
        return 2.0 * integral / self.b / self.tau

    @staticmethod
    def filter_otm_quotes(atm_strike: float, strikes: np.array, quotes: np.array, opt_type: OptionType):
        idx_perm = strikes.argsort()
        strikes = strikes[idx_perm]
        quotes = quotes[idx_perm]
        if opt_type == OptionType.put:
            indicators_otm = strikes < atm_strike
        else:
            indicators_otm = strikes > atm_strike
        strikes_otm = strikes[indicators_otm]
        quotes_otm = quotes[indicators_otm]
        return strikes_otm, quotes_otm