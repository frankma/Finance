from math import sqrt, log
from scipy.stats import norm
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class NormalModel(object):

    @staticmethod
    def calc_d(f: float, k: float, tau: float, sig: float):
        return (f - k) / (sig * sqrt(tau))

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d = NormalModel.calc_d(f, k, tau, sig)
        return b * (eta * (f - k) * norm.cdf(eta * d) + sig * sqrt(tau) * norm.pdf(d))

    @staticmethod
    def imp_vol(f: float, k: float, tau: float, price: float, b: float, opt_type: OptionType):
        vol = 0.88 * f  # initial guess
        v = NormalModel.price(f, k, tau, vol, b, opt_type)

        count = 1

        while abs(v - price) > 1e-12 and count < 99:
            vega = NormalModel.vega(f, k, vol, vol, b)
            vol += (price - v) / vega
            v = NormalModel.price(f, k, tau, vol, b, opt_type)
            count += 1

        if count > 99:
            print('WARNING: black vol searching max out iterations.')

        return vol

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d = NormalModel.calc_d(f, k, tau, sig)
        return b * eta * norm.cdf(eta * d)

    @staticmethod
    def vega(f: float, k: float, tau: float, sig: float, b: float):
        d = NormalModel.calc_d(f, k, tau, sig)
        return b * sqrt(tau) * norm.pdf(d)

    @staticmethod
    def theta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        d = NormalModel.calc_d(f, k, tau, sig)
        r = -log(b) / tau
        v = NormalModel.price(f, k, tau, sig, b, opt_type)
        return -r * b * v + 0.5 * b * sig / sqrt(tau) * norm.pdf(d)

    @staticmethod
    def gamma(f: float, k: float, tau: float, sig: float, b: float):
        d = NormalModel.calc_d(f, k, tau, sig)
        return b * norm.pdf(d) / sig / sqrt(tau)
