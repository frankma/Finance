from math import log, sqrt
import numpy as np
from scipy.stats import norm
from src.Utils.OptionType import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson

__author__ = 'frank.ma'


class Black76(object):
    @staticmethod
    def calc_d1(f: float, k: float, tau: float, sig: float):
        return (log(f / k) + 0.5 * sig ** 2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def calc_d2(f: float, k: float, tau: float, sig: float):
        return (log(f / k) - 0.5 * sig ** 2 * tau) / (sig * sqrt(tau))

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.calc_d1(f, k, tau, sig)
        d2 = Black76.calc_d2(f, k, tau, sig)
        return b * eta * (f * norm.cdf(eta * d1) - k * norm.cdf(eta * d2))

    @staticmethod
    def imp_vol(f: float, k: float, tau: float, price: float, b: float, opt_type: OptionType, method='Brent'):

        class PriceFunction(IUnivariateFunction):
            def evaluate(self, x):
                return Black76.price(f, k, tau, x, b, opt_type) - price

        class VegaFunction(IUnivariateFunction):
            def evaluate(self, x):
                return Black76.vega(f, k, tau, x, b)

        pf = PriceFunction()
        vf = VegaFunction()

        if method == 'Brent':
            bt = Brent(pf, 1e-4, 10.0)
            vol = bt.solve()
        elif method == 'Newton-Raphson':
            nr = NewtonRaphson(pf, vf, 0.88)
            vol = nr.solve()
        else:
            raise ValueError('Unrecognized optimization method %s.' % method)

        return vol

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.calc_d1(f, k, tau, sig)
        return b * eta * norm.cdf(eta * d1)

    @staticmethod
    def delta_k(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = Black76.calc_d2(f, k, tau, sig)
        return -b * eta * norm.cdf(eta * d2)

    @staticmethod
    def vega(f: float, k: float, tau: float, sig: float, b: float):
        d1 = Black76.calc_d1(f, k, tau, sig)
        return b * f * sqrt(tau) * norm.pdf(d1)

    @staticmethod
    def theta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        d1 = Black76.calc_d1(f, k, tau, sig)
        price = Black76.price(f, k, tau, sig, b, opt_type)
        r = -log(b) / tau
        return -b * f * norm.pdf(d1) * sig / 2. / sqrt(tau) + r * price

    @staticmethod
    def theta_s(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = Black76.calc_d1(f, k, tau, sig)
        d2 = Black76.calc_d2(f, k, tau, sig)
        r = -log(b) / tau
        q = log(k / f) / tau + r
        return -b * f * norm.pdf(d1) * sig / 2. / sqrt(tau) + b * eta * \
                                                              (q * f * norm.cdf(eta * d1) - r * k * norm.cdf(eta * d2))

    @staticmethod
    def rho(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        price = Black76.price(f, k, tau, sig, b, opt_type)
        return -tau * b * eta * price

    @staticmethod
    def rho_s(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = Black76.calc_d2(f, k, tau, sig)
        return tau * b * eta * k * norm.cdf(eta * d2)

    @staticmethod
    def gamma(f: float, k: float, tau: float, sig: float, b: float):
        d1 = Black76.calc_d1(f, k, tau, sig)
        return b * norm.pdf(d1) / f / sig / sqrt(tau)

    @staticmethod
    def gamma_k(f: float, k: float, tau: float, sig: float, b: float):
        d2 = Black76.calc_d2(f, k, tau, sig)
        return b * norm.pdf(d2) / k / sig / sqrt(tau)


class Black76Vec(Black76):
    @staticmethod
    def calc_d1(f: np.array, k: np.array, tau: float, sig: np.array) -> np.array:
        return np.log(f / k) / sig / sqrt(tau) + 0.5 * sig * np.sqrt(tau)

    @staticmethod
    def calc_d2(f: np.array, k: np.array, tau: float, sig: np.array) -> np.array:
        return np.log(f / k) / sig / sqrt(tau) - 0.5 * sig * np.sqrt(tau)

    @staticmethod
    def price(f: np.array, k: np.array, tau: float, sig: np.array, b: float, opt_type: OptionType) -> np.array:
        eta = opt_type.value
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        d2 = Black76Vec.calc_d2(f, k, tau, sig)
        return b * eta * (f * norm.cdf(eta * d1) - k * norm.cdf(eta * d2))

    @staticmethod
    def imp_vol(f: np.array, k: np.array, tau: float, price: np.array, b: float, opt_type: OptionType,
                method='Brent') -> np.array:
        if k.__len__() != price.__len__():
            raise ValueError('expect strike and price in the same length')
        vols = np.zeros(k.__len__())
        for idx, p in enumerate(price):
            vols[idx] = Black76.imp_vol(f, k[idx], tau, p, b, opt_type, method=method)
        return vols

    @staticmethod
    def delta(f: np.array, k: np.array, tau: float, sig: np.array, b: float, opt_type: OptionType) -> np.array:
        eta = opt_type.value
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        return eta * b * norm.pdf(eta * d1)

    @staticmethod
    def delta_k(f: np.array, k: np.array, tau: float, sig: np.array, b: float, opt_type: OptionType) -> np.array:
        eta = opt_type.value
        d2 = Black76Vec.calc_d2(f, k, tau, sig)
        return -eta * b * norm.pdf(eta * d2)

    @staticmethod
    def vega(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        return b * f * norm.pdf(d1) * sqrt(tau)

    @staticmethod
    def gamma(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        return b * norm.pdf(d1) / f / sig / sqrt(tau)

    @staticmethod
    def gamma_k(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        d2 = Black76Vec.calc_d2(f, k, tau, sig)
        return b * norm.pdf(d2) / k / sig / sqrt(tau)

    @staticmethod
    def vomma(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        d2 = Black76Vec.calc_d2(f, k, tau, sig)
        return vega * d1 * d2 / sig

    @staticmethod
    def vanna(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d2 = Black76Vec.calc_d2(f, k, tau, sig)
        return -vega * d2 / sig / sqrt(tau) / f

    @staticmethod
    def vanna_k(f: np.array, k: np.array, tau: float, sig: np.array, b: float) -> np.array:
        vega = Black76Vec.vega(f, k, tau, sig, b)
        d1 = Black76Vec.calc_d1(f, k, tau, sig)
        return vega * d1 / sig / sqrt(tau) / k
