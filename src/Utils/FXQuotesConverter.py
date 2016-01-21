import logging

import numpy as np
from scipy.stats import norm

from src.Utils.OptionType import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class FXQuotesConverter(object):
    def __init__(self, spot: float, tau: float, rate_dom: float, rate_for: float, quotes: dict,
                 is_premium_adj: bool = False, is_forward_delta: bool = True):
        self.tau = tau
        self.spot = spot
        self.rate_dom = rate_dom
        self.rate_for = rate_for
        self.quotes = quotes
        self.strikes = None
        self.vols = None
        self.is_premium_adj = is_premium_adj
        self.is_forward_delta = is_forward_delta

    def convert(self, method: str = 'Newton-Raphson'):
        keys = ['rr_10', 'rr_25', 'atm_50', 'sm_25', 'sm_10']
        deltas = [-0.1, -0.25, 0.5, 0.25, 0.1]
        is_atm = [False, False, True, False, False]
        if not all(key in self.quotes for key in keys):
            msg = 'all keys of quote set %s are required' % keys.__str__()
            logger.error(msg)
            raise ValueError(msg)

        vols = self.read_quotes(self.quotes)
        ks = np.zeros(np.shape(vols), dtype=float)
        for kdx, delta in enumerate(deltas):
            ks[kdx] = self.vol_to_strike(vols[kdx], delta, self.tau, self.spot, self.rate_dom, self.rate_for,
                                         is_atm[kdx], is_premium_adj=self.is_premium_adj,
                                         is_forward_delta=self.is_forward_delta, method=method)
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
    def vol_to_strike(sig: float, delta: float, tau: float, spot: float, rate_dom: float, rate_for: float, is_atm: bool,
                      is_premium_adj: bool = False, is_forward_delta: bool = True, method='Newton-Raphson'):
        opt_type = OptionType.call if delta > 0.0 else OptionType.put
        eta = float(opt_type.value)
        bond_for = np.exp(-rate_for * tau)

        if is_premium_adj:
            # in case of premium adjusted, no analytical solution can be easily found hence solve it numerically

            class _ZerothFunc(IUnivariateFunction):

                def evaluate(self, x):
                    d2 = BSM.calc_d2(spot, x, tau, rate_dom, rate_for, sig)
                    delta_tgt = eta * norm.cdf(eta * d2)
                    if not is_atm:
                        delta_tgt *= x / spot
                    if not is_forward_delta:
                        delta_tgt *= bond_for
                    return delta_tgt - delta

            class _FirstFunc(IUnivariateFunction):

                def evaluate(self, x):
                    d2 = BSM.calc_d2(spot, x, tau, rate_dom, rate_for, sig)

                    if is_atm:
                        first_derivative = - norm.pdf(d2) / sig / np.sqrt(tau) / x
                    else:
                        first_derivative = (eta * norm.cdf(eta * d2) - norm.pdf(d2) / sig / np.sqrt(tau)) / spot

                    if not is_forward_delta:
                        first_derivative *= bond_for

                    return first_derivative

            zeroth = _ZerothFunc()
            first = _FirstFunc()

            if method == 'Brent':
                bt = Brent(zeroth, 1e-4 * spot, 10.0 * spot)
                strike = bt.solve()
            elif method == 'Newton-Raphson':
                nr = NewtonRaphson(zeroth, first, spot)
                strike = nr.solve()
            else:
                msg = 'unrecognized optimization method %s.' % method
                logger.error(msg)
                raise ValueError(msg)

        else:
            if is_forward_delta:
                rev_term = eta * norm.ppf(eta * delta) * sig * np.sqrt(tau)
            else:
                rev_term = eta * norm.ppf(eta * delta / bond_for) * sig * np.sqrt(tau)
            strike = spot / np.exp(rev_term - (rate_dom - rate_for + 0.5 * (sig ** 2)) * tau)

        return strike
