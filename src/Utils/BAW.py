import numpy as np

from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson

__author__ = 'frank.ma'


class BAW(object):
    @staticmethod
    def __calc_lam(tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        sig_sq = sig ** 2
        m = 2.0 * r / sig_sq
        n = 2.0 * (r - q) / sig_sq
        n_min_one = n - 1.0
        g = 1.0 - np.exp(-r * tau)
        lam = 0.5 * (-n_min_one + eta * np.sqrt((n_min_one ** 2) + 4.0 * m / g))
        return lam

    @staticmethod
    def calc_s_optimum(k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        lam = BAW.__calc_lam(tau, r, q, sig, opt_type)

        class FFunction(IUnivariateFunction):
            def evaluate(self, x):
                price = BSM.price(x, k, tau, r, q, sig, opt_type)
                delta = BSM.delta(x, k, tau, r, q, sig, opt_type)
                return eta * (x - k) - price - (eta - delta) * x / lam

        class FPrimFunction(IUnivariateFunction):
            def evaluate(self, x):
                delta = BSM.delta(x, k, tau, r, q, sig, opt_type)
                gamma = BSM.gamma(x, k, tau, r, q, sig)
                return (eta - delta) * (1.0 - 1.0 / lam) + gamma * x / lam

        f = FFunction()
        f_prime = FPrimFunction()

        nr = NewtonRaphson(f, f_prime, k)
        nr.ABS_TOL = 1e-6
        s_opt = nr.solve()

        return s_opt

    @staticmethod
    def price(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_optimum = BAW.calc_s_optimum(k, tau, r, q, sig, opt_type)
        if eta * s < eta * s_optimum:
            lam = BAW.__calc_lam(tau, r, q, sig, opt_type)
            delta_optimum = BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)
            alpha = (eta - delta_optimum) / lam
            price_ame = BSM.price(s, k, tau, r, q, sig, opt_type) + alpha * s_optimum * ((s / s_optimum) ** lam)
        else:
            price_ame = eta * (s - k)
        return price_ame

    @staticmethod
    def delta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType, ds: float = 1e-6):
        v_d = BAW.price(s * (1.0 - ds), k, tau, r, q, sig, opt_type)
        v_u = BAW.price(s * (1.0 + ds), k, tau, r, q, sig, opt_type)
        return (v_u - v_d) / (2.0 * s * ds)

    @staticmethod
    def gamma(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType, ds: float = 1e-6):
        v_d = BAW.price(s * (1.0 - ds), k, tau, r, q, sig, opt_type)
        v = BAW.price(s, k, tau, r, q, sig, opt_type)
        v_u = BAW.price(s * (1.0 + ds), k, tau, r, q, sig, opt_type)
        return (v_u - 2.0 * v + v_d) / ((s * ds) ** 2)
