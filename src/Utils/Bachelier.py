from math import sqrt
from scipy.stats import norm
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class Bachelier(object):

    @staticmethod
    def calc_d(f: float, k: float, tau: float, sig: float):
        return (f - k) / (sig * sqrt(tau))

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d = Bachelier.calc_d(f, k, tau, sig)
        return b * (eta * (f - k) * norm.cdf(eta * d) + sig * sqrt(tau) * norm.pdf(d))

    @staticmethod
    def normal_vol(f: float, k:float, tau: float, price: float, b: float, opt_type: OptionType):
        # TODO: need vega for downhill calculation
        pass

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        pass