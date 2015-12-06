import numpy as np

__author__ = 'frank.ma'


class FXQuotesConverter(object):
    def __init__(self, tau: float, spot: float, rate_dom: float, rate_for: float, quotes: dict):
        self.tau = tau
        self.spot = spot
        self.rate_dom = rate_dom
        self.rate_for = rate_for
        self.quotes = quotes

    @staticmethod
    def read_quotes(quotes):
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
        vols_vec = np.linalg.solve(mtx, quotes_vec)
        return vols_vec
