import numpy as np

from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson

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

    @staticmethod
    def vol_to_strike(sig: float, delta: float, tau: float, spot: float, rate_dom: float, rate_for: float,
                      method='Newton-Raphson'):
        opt_type = OptionType.call if delta > 0.0 else OptionType.put

        class DeltaFunction(IUnivariateFunction):
            def evaluate(self, x):
                return BSM.delta(spot, x, tau, rate_dom, rate_for, sig, opt_type) - delta

        class DDeltaDKFunction(IUnivariateFunction):
            def evaluate(self, x):
                gamma = BSM.gamma(spot, x, tau, rate_dom, rate_for, sig)
                return -gamma * spot / x  # d^2 v / ds^2 is close to d^2 v / dk ds with correction term - s / x

        zeroth = DeltaFunction()
        first = DDeltaDKFunction()

        if method == 'Brent':
            bt = Brent(zeroth, 1e-4 * spot, 10.0 * spot)
            strike = bt.solve()
        elif method == 'Newton-Raphson':
            nr = NewtonRaphson(zeroth, first, spot)
            strike = nr.solve()
        else:
            raise ValueError('Unrecognized optimization method %s.' % method)

        return strike
