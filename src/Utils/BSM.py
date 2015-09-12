from scipy.stats import norm
from src.Utils import OptionType
from math import log, exp, sqrt
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
    def imp_vol(s: float, k: float, tau: float, r: float, q: float, price: float, opt_type: OptionType):

        class PriceFunction(IUnivariateFunction):

            def evaluate(self, x):
                return BSM.price(s, k, tau, r, q, x, opt_type) - price

        class VegaFunction(IUnivariateFunction):

            def evaluate(self, x):
                return BSM.vega(s, k, tau, r, q, x)

        pf = PriceFunction()
        vf = VegaFunction()

        nr = NewtonRaphson(pf, vf, 0.88)
        vol = nr.solve()

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
        term1 = -exp(-q * tau) * s * norm.pdf(d1) * sig / 2 / sqrt(tau)
        term2 = eta * q * s * exp(-q * tau) * norm.cdf(eta * d1)
        term3 = - eta * r * k * exp(-r * tau) * norm.cdf(eta * d2)
        return term1 * term2 * term3

    @staticmethod
    def rho(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        d2 = BSM.calc_d2(s, k, tau, r, q, sig)
        return eta * tau * exp(-r * tau) * norm.cdf(eta * d2)