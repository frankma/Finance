import numpy as np

from src.Utils import OptionType
from src.Utils.BSM import BSMVec

__author__ = 'frank.ma'


class SingleVariableDeltaHedgingValuator(object):
    def __init__(self, k: float, r: float, q: float, sig: float, opt_type: OptionType, model='lognormal'):
        self.k = k
        self.r = r
        self.q = q
        self.sig = sig
        self.opt_type = opt_type
        self.model = model

    def price(self, s: np.array, tau: float):
        return BSMVec.price(s, self.k, tau, self.r, self.q, self.sig, self.opt_type)

    def payoff(self, s: np.array):
        return BSMVec.payoff(s, self.k, self.opt_type)

    def delta(self, s: np.array, tau: float):
        return BSMVec.delta(s, self.k, tau, self.r, self.q, self.sig, self.opt_type)

    def gamma(self, s: np.array, tau: float):
        return BSMVec.gamma(s, self.k, tau, self.r, self.q, self.sig)


class SingleVariableDeltaHedgingValuatorBasket(object):
    def __init__(self, valuators: list):
        for valuator in valuators:
            isinstance(valuators, SingleVariableDeltaHedgingValuator)
        self.valuators = valuators

    def price(self, s: np.array, tau: float):
        v = np.zeros(s.__len__())
        for valuator in self.valuators:
            v += valuator.price(s, tau)
        return v

    def delta(self, s: np.array, tau: float):
        d = np.zeros(s.__len__())
        for valuator in self.valuators:
            d += valuator.delta(s, tau)
        return d

    def gamma(self, s: np.array, tau: float):
        g = np.zeros(s.__len__())
        for valuator in self.valuators:
            g += valuator.gamma(s, tau)
        return g
