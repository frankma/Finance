from math import sqrt, log

from scipy.stats import norm

from src.Utils.OptionType import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson

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
    def imp_vol(f: float, k: float, tau: float, price: float, b: float, opt_type: OptionType, method='Brent'):

        class PriceFunction(IUnivariateFunction):

            def evaluate(self, x: float):
                return NormalModel.price(f, k, tau, x, b, opt_type) - price

        class VegaFunction(IUnivariateFunction):

            def evaluate(self, x):
                return NormalModel.vega(f, k, tau, x, b)

        pf = PriceFunction()
        vf = VegaFunction()

        if method == 'Brent':
            bt = Brent(pf, 1e-4, 10.0 * f)
            vol = bt.solve()
        elif method == 'Newton-Raphson':
            nr = NewtonRaphson(pf, vf, 0.88 * f)
            vol = nr.solve()
        else:
            raise Exception('Unrecognized optimization method %s.' % method)

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
