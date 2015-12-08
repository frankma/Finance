import numpy as np
from scipy.stats import norm

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class FXQuotesConverter(object):
    def __init__(self, spot: float, tau: float, rate_dom: float, rate_for: float, quotes: dict):
        self.tau = tau
        self.spot = spot
        self.rate_dom = rate_dom
        self.rate_for = rate_for
        self.quotes = quotes
        self.strikes = None
        self.vols = None

    def convert(self, method: str = 'Newton-Raphson'):
        keys = ['rr_10', 'rr_25', 'atm_50', 'sm_25', 'sm_10']
        deltas = [-0.1, -0.25, 0.5, 0.25, 0.1]
        if not all(key in self.quotes for key in keys):
            raise ValueError('all keys of quote set %s are required' % keys.__str__())

        vols = self.read_quotes(self.quotes)
        ks = np.zeros(np.shape(vols), dtype=float)
        for kdx, delta in enumerate(deltas):
            ks[kdx] = self.vol_to_strike(vols[kdx], delta, self.tau, self.spot, self.rate_dom, self.rate_for,
                                         method=method)
        self.strikes = ks
        self.vols = vols
        return ks, vols

    @staticmethod
    def read_quotes(quotes, seven_quotes: bool = False):
        rr_10 = quotes['rr_10']
        rr_25 = quotes['rr_25']
        atm = quotes['atm_50']
        sm_25 = quotes['sm_25']
        sm_10 = quotes['sm_10']

        mtx = np.array([[-1.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.5, -1.0, 0.5, 0.0],
                        [0.5, 0.0, -1.0, 0.0, 0.5]])
        quotes_vec = np.array([rr_10, rr_25, atm, sm_25, sm_10])

        if seven_quotes:
            mtx = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.5, -1.0, 0.5, 0.0, 0.0],
                            [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0],
                            [0.5, 0.0, 0.0, -1.0, 0.0, 0.0, 0.5]])
            rr_05 = quotes['rr_05']
            sm_05 = quotes['sm_05']
            quotes_vec = np.array([rr_05, rr_10, rr_25, atm, sm_25, sm_10, sm_05])

        vols_vec = np.linalg.solve(mtx, quotes_vec)
        return vols_vec

    @staticmethod
    def vol_to_strike(sig: float, forward_delta: float, tau: float, spot: float, rate_dom: float, rate_for: float,
                      is_premium_adj: bool = False, method='Newton-Raphson'):
        if is_premium_adj:
            # TODO: make a proper method search for this
            raise NotImplementedError('not implemented yet')
        else:
            opt_type = OptionType.call if forward_delta > 0.0 else OptionType.put
            eta = float(opt_type.value)
            strike = spot / np.exp(eta * norm.ppf(eta * forward_delta) * sig * np.sqrt(tau)
                                   - (rate_dom - rate_for + 0.5 * (sig ** 2)) * tau)

        return strike
