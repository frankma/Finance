from math import log, exp, sqrt

import numpy as np
from scipy.stats import norm

from src.Utils import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson

__author__ = 'frank.ma'


class BSM(object):

    @staticmethod
    def calc_d1(s: float, k: float, tau: float, r: float, q: float, sig: float):
        return (log(s / k) + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt(tau))

    @staticmethod
    def calc_d2(s: float, k: float, tau: float, r: float, q: float, sig: float):
        return (log(s / k) + (r - q - 0.5 * sig**2) * tau) / (sig * sqrt(tau))

    @staticmethod
    def price(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        d2 = BSM.calc_d2(s, k, tau, r, q, sig)
        return eta * (exp(-q * tau) * s * norm.cdf(eta * d1) - exp(-r * tau) * k * norm.cdf(eta * d2))

    @staticmethod
    def imp_vol(s: float, k: float, tau: float, r: float, q: float, price: float, opt_type: OptionType, method='Brent'):

        class PriceFunction(IUnivariateFunction):

            def evaluate(self, x):
                return BSM.price(s, k, tau, r, q, x, opt_type) - price

        class VegaFunction(IUnivariateFunction):

            def evaluate(self, x):
                return BSM.vega(s, k, tau, r, q, x)

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
    def delta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        return eta * norm.cdf(eta * d1)

    @staticmethod
    def gamma(s: float, k: float, tau: float, r: float, q: float, sig: float):
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        return exp(-q * tau) * norm.pdf(d1) / s / sig / sqrt(tau)

    @staticmethod
    def vega(s: float, k: float, tau: float, r: float, q: float, sig: float):
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        return s * exp(-q * tau) * norm.pdf(d1) * sqrt(tau)

    @staticmethod
    def theta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d1 = BSM.calc_d1(s, k, tau, r, q, sig)
        d2 = BSM.calc_d2(s, k, tau, r, q, sig)
        term1 = -exp(-q * tau) * s * norm.pdf(d1) * sig / 2.0 / sqrt(tau)
        term2 = eta * q * s * exp(-q * tau) * norm.cdf(eta * d1)
        term3 = - eta * r * k * exp(-r * tau) * norm.cdf(eta * d2)
        return term1 * term2 * term3

    @staticmethod
    def rho(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = BSM.calc_d2(s, k, tau, r, q, sig)
        return eta * tau * exp(-r * tau) * norm.cdf(eta * d2)


class BSMVecS(BSM):

    @staticmethod
    def calc_d1(s: np.array, k: float, tau: float, r: float, q: float, sig: float):
        return (np.log(s / k) + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt(tau))

    @staticmethod
    def calc_d2(s: np.array, k: float, tau: float, r: float, q: float, sig: float):
        return (np.log(s / k) + (r - q - 0.5 * sig**2) * tau) / (sig * sqrt(tau))

    @staticmethod
    def price(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        if tau < 1e-6:
            if tau < 0.0:
                print('WARNING: negative tau %r is provided in pricing function, return payoff.' % tau)
            return BSMVecS.payoff(s, k, opt_type)
        else:
            eta = opt_type.value
            d_1 = BSMVecS.calc_d1(s, k, tau, r, q, sig)
            d_2 = BSMVecS.calc_d2(s, k, tau, r, q, sig)
            return eta * (exp(-q * tau) * s * norm.cdf(eta * d_1) - exp(-r * tau) * k * norm.cdf(eta * d_2))

    @staticmethod
    def payoff(s: np.array, k: float, opt_type: OptionType):
        eta = opt_type.value
        return np.maximum(eta * (s - k), np.zeros(s.__len__()))

    @staticmethod
    def delta(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        if tau < 1e-6:
            if tau < 0.0:
                print('WARNING: negative tau %r is provided in delta function, return zeros.' % tau)
            return np.zeros(s.__len__())
        else:
            eta = opt_type.value
            d_1 = BSMVecS.calc_d1(s, k, tau, r, q, sig)
            return eta * exp(-q * tau) * norm.cdf(eta * d_1)

    @staticmethod
    def gamma(s: np.array, k: float, tau: float, r: float, q: float, sig: float):
        d_1 = BSMVecS.calc_d1(s, k, tau, r, q, sig)
        return exp(-q * tau) * norm.pdf(d_1) / (s * sig * sqrt(tau))

    @staticmethod
    def vega(s: np.array, k: float, tau: float, r: float, q: float, sig: float):
        d_1 = BSMVecS.calc_d1(s, k, tau, r, q, sig)
        return s * exp(-q * tau) * norm.pdf(d_1) * sqrt(tau)

    @staticmethod
    def theta(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d_1 = BSMVecS.calc_d1(s, k, tau, r, q, sig)
        d_2 = BSMVecS.calc_d2(s, k, tau, r, q, sig)
        term1 = -exp(-q * tau) * s * norm.pdf(d_1) / (2.0 * sqrt(tau))
        term2 = eta * q * s * exp(-q * tau) * norm.cdf(eta * d_1)
        term3 = eta * r * k * exp(-r * tau) * norm.cdf(eta * d_2)
        return term1 * term2 * term3

    @staticmethod
    def rho(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d_2 = BSMVecS.calc_d2(s, k, tau, r, q, sig)
        return eta * k * tau * exp(-r * tau) * norm.cdf(eta * d_2)
