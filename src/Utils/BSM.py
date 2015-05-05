from scipy.stats import norm
from src.Utils import OptionType
from math import log, exp, sqrt

__author__ = 'frank.ma'


class BSM(object):

    @staticmethod
    def calc_d1(s: float, k: float, tau: float, r: float, q: float, sig: float):
        return (log(s / k) + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt(tau))

    @staticmethod
    def calc_d2(s: float, k: float, tau: float, r: float, q: float, sig: float):
        return BSM.calc_d1(s, k, tau, r, q, sig) - sig * sqrt(tau)

    @staticmethod
    def price(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        d2 = BSM.calc_d2(s, k, tau, r, q, sig)
        return eta * (exp(-q * tau) * s * norm.cdf(eta * d1) - exp(-r * tau) * k * norm.cdf(eta * d2))

    @staticmethod
    def delta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        return eta * norm.cdf(eta * d1)