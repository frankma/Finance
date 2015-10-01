import numpy as np

from src.Utils import OptionType
from src.Utils.BSM import BSMVecS

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
        return BSMVecS.price(s, self.k, tau, self.r, self.q, self.sig, self.opt_type)

    def payoff(self, s: np.array):
        return BSMVecS.payoff(s, self.k, self.opt_type)

    def delta(self, s: np.array, tau: float):
        return BSMVecS.delta(s, self.k, tau, self.r, self.q, self.sig, self.opt_type)

    def gamma(self, s: np.array, tau):
        return BSMVecS.gamma(s, self.k, tau, self.r, self.q, self.sig)
