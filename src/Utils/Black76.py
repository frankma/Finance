from math import log, sqrt
from scipy.stats import norm
from src.Utils.OptionType import OptionType


__author__ = 'frank.ma'


class Black76(object):

    @staticmethod
    def cpt_d1(f: float, k: float, tau: float, sig: float):
        return (log(f / k) + 0.5 * sig**2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def cpt_d2(f: float, k: float, tau: float, sig: float):
        return (log(f / k) - 0.5 * sig**2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.cpt_d1(f, k, tau, sig)
        d2 = Black76.cpt_d2(f, k, tau, sig)
        return b * eta * (f * norm.cdf(eta * d1) - k * norm.cdf(eta * d2))

    @staticmethod
    def imp_vol(f: float, k: float, tau: float, price: float, b: float, opt_type: OptionType):
        vol = 0.88  # initial guess
        v = Black76.price(f, k, tau, vol, b, opt_type)

        count = 0

        while abs(v / price - 1) > 1e-6 and count < 100:
            vega = Black76.vega(f, k, tau, vol, b)
            vol += (price - v) / vega
            v = Black76.price(f, k, tau, vol, b, opt_type)
            count += 1

        if count > 99:
            print('WARNING: black vol searching max out iterations.')

        return vol

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.cpt_d1(f, k, tau, sig)
        return b * eta * norm.cdf(eta * d1)

    @staticmethod
    def delta_k(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = Black76.cpt_d2(f, k, tau, sig)
        return -b * eta * norm.cdf(eta * d2)

    @staticmethod
    def vega(f: float, k: float, tau: float, sig: float, b: float):
        d1 = Black76.cpt_d1(f, k, tau, sig)
        return b * f * sqrt(tau) * norm.pdf(d1)

    @staticmethod
    def theta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        d1 = Black76.cpt_d1(f, k, tau, sig)
        price = Black76.price(f, k, tau, sig, b, opt_type)
        r = -log(b) / tau
        return -b * f * norm.pdf(d1) * sig / 2. / sqrt(tau) + r * price

    @staticmethod
    def theta_s(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.cpt_d1(f, k, tau, sig)
        d2 = Black76.cpt_d2(f, k, tau, sig)
        r = -log(b) / tau
        q = log(k / f) / tau + r
        return -b * f * norm.pdf(d1) * sig / 2. / sqrt(tau) + b * eta * (q * f * norm.cdf(eta * d1)
                                                                         - r * k * norm.cdf(eta * d2))

    @staticmethod
    def rho(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        price = Black76.price(f, k, tau, sig, b, opt_type)
        return -tau * b * eta * price

    @staticmethod
    def rho_s(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = Black76.cpt_d2(f, k, tau, sig)
        return tau * b * eta * k * norm.cdf(eta * d2)

    @staticmethod
    def gamma(f: float, k: float, tau: float, sig: float, b: float):
        d1 = Black76.cpt_d1(f, k, tau, sig)
        return b * norm.pdf(d1) / f / sig / sqrt(tau)

    @staticmethod
    def gamma_k(f: float, k: float, tau: float, sig: float, b: float):
        d2 = Black76.cpt_d2(f, k, tau, sig)
        return b * norm.pdf(d2) / k / sig / sqrt(tau)