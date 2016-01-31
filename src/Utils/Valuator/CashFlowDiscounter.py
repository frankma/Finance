import logging

import numpy as np

from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class CashFlowDiscounter(object):
    def __init__(self, taus: np.array, cash_flows: np.array):
        self.ts = taus
        self.cs = cash_flows

    @staticmethod
    def calc_present_values(taus: np.array, cash_flows: np.array, rates: np.array):
        return np.exp(-taus * rates) * cash_flows

    def calc_npv(self, rates: float or np.array):
        cash_flows_discounted = self.calc_present_values(self.ts, self.cs, rates)
        return np.sum(cash_flows_discounted)

    def calc_irr(self, lb: float = -0.1, ub: float = 2.0):
        # this is an internal wrapper class for the root search
        class _NPV(IUnivariateFunction):
            def __init__(self, taus: np.array, cash_flows: np.array):
                self.taus = taus
                self.cash_flows = cash_flows
                pass

            def evaluate(self, x):
                return np.sum(CashFlowDiscounter.calc_present_values(self.taus, self.cash_flows, x))

        tgt_func = _NPV(self.ts, self.cs)
        solver = Brent(tgt_func, lb, ub)
        irr = solver.solve()

        return irr
