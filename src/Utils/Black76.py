from enum import Enum
from math import log, sqrt
from scipy.stats import norm

__author__ = 'frank.ma'


class OptionType(Enum):
    call = 1
    put = -1


class Black76(object):

    @staticmethod
    def cpt_d_1(f: float, k: float, tau: float, sig: float):
        return (log(f / k) + 0.5 * sig**2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def cpt_d_2(f: float, k: float, tau: float, sig: float):
        return (log(f / k) - 0.5 * sig**2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        d_2 = Black76.cpt_d_2(f, k, tau, sig)
        return b * eta.value * (f * norm.cdf(eta.value * d_1) - k * norm.cdf(eta.value * d_2))

    @staticmethod
    def imp_vol(f: float, k: float, tau: float, price: float, b: float, eta: OptionType):
        vol = 0.88  # initial guess
        v = Black76.price(f, k, tau, vol, b, eta)

        count = 0

        while abs(v / price - 1) > 1e-6 and count < 100:
            vega = Black76.vega(f, k, tau, vol, b)
            vol += (price - v) / vega
            v = Black76.price(f, k, tau, vol, b, eta)
            count += 1

        if count > 99:
            print('WARNING: black vol searching max out iterations.')

        return vol

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        return b * eta.value * norm.cdf(eta.value * d_1)

    @staticmethod
    def delta_k(f: float, k: float, tau: float, sig:float, b: float, eta: OptionType):
        d_2 = Black76.cpt_d_2(f, k, tau, sig)
        return -b * eta.value * norm.cdf(eta.value * d_2)

    @staticmethod
    def vega(f: float, k: float, tau: float, sig: float, b: float):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        return b * f * sqrt(tau) * norm.pdf(d_1)

    @staticmethod
    def theta(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        price = Black76.price(f, k, tau, sig, b, eta)
        r = -log(b) / tau
        return -b * f * norm.pdf(d_1) * sig / 2. / sqrt(tau) + r * price

    @staticmethod
    def theta_s(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        d_2 = Black76.cpt_d_2(f, k, tau, sig)
        r = -log(b) / tau
        q = log(k / f) / tau + r
        return -b * f * norm.pdf(d_1) * sig / 2. / sqrt(tau) + b * eta.value * (q * f * norm.cdf(eta.value * d_1)
                                                                                - r * k * norm.cdf(eta.value * d_2))

    @staticmethod
    def rho(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        price = Black76.price(f, k, tau, sig, b, eta)
        return -tau * b * eta.value * price

    @staticmethod
    def rho_s(f: float, k: float, tau: float, sig: float, b: float, eta: OptionType):
        d_2 = Black76.cpt_d_2(f, k, tau, sig)
        return tau * b * eta.value * k * norm.cdf(eta.value * d_2)

    @staticmethod
    def gamma(f: float, k: float, tau: float, sig: float, b: float):
        d_1 = Black76.cpt_d_1(f, k, tau, sig)
        return b * norm.pdf(d_1) / f / sig / sqrt(tau)

    @staticmethod
    def gamma_k(f: float, k: float, tau: float, sig: float, b: float):
        d_2 = Black76.cpt_d_2(f, k, tau, sig)
        return b * norm.pdf(d_2) / k / sig / sqrt(tau)