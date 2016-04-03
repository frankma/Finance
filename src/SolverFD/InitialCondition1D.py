import logging

import numpy as np
from src.Utils.PayoffType import PayoffType

from src.Utils.Types.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class InitialCondition1D(object):
    def __init__(self):
        pass

    def get_state(self, xs: np.array):
        pass


class InitialCondition1DVanilla(InitialCondition1D):
    def __init__(self, strike: float, opt_type: OptionType, payoff_type: PayoffType):
        super().__init__()
        self.strike = strike
        self.opt_type = opt_type
        self.payoff_type = payoff_type

    def get_state(self, xs: np.array):
        zeros = np.zeros(xs.shape, dtype=float)
        eta = float(self.opt_type.value)
        if self.payoff_type == PayoffType.European or self.payoff_type == PayoffType.American:
            payoff = np.maximum(eta * (xs - self.strike), zeros)
        elif self.payoff_type == PayoffType.Binary:
            payoff = np.ones(xs.shape, dtype=float)
            payoff[(eta * (xs - self.strike)) <= zeros] = 0.0
        elif self.payoff_type == PayoffType.CashOrNothing:
            payoff = np.full(xs.shape, self.strike)
            payoff[(eta * (xs - self.strike)) <= zeros] = 0.0
        elif self.payoff_type == PayoffType.AssetOrNothing:
            payoff = xs.copy()
            payoff[(eta * (xs - self.strike)) <= zeros] = 0.0
        else:
            raise NotImplementedError('payoff type (%s) is not implemented' % self.payoff_type.__str__())

        return payoff
