import logging

import numpy as np

from src.Utils.OptionType import OptionType
from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction
from src.Utils.Solver.NewtonRaphson import NewtonRaphson
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class BAW(object):
    @staticmethod
    def calc_lambda(tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        sig_sq = sig ** 2
        beta_0 = 2.0 * r / sig_sq
        beta_1 = 2.0 * (r - q) / sig_sq
        beta_1_min_one = beta_1 - 1.0
        g = 1.0 - np.exp(-r * tau)
        lam = 0.5 * (-beta_1_min_one + eta * np.sqrt((beta_1_min_one ** 2) + 4.0 * beta_0 / g))
        return lam

    @staticmethod
    def calc_s_optimum(k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        lam = BAW.calc_lambda(tau, r, q, sig, opt_type)

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
            lam = BAW.calc_lambda(tau, r, q, sig, opt_type)
            delta_optimum = BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)
            alpha = (eta - delta_optimum) / lam
            price_ame = BSM.price(s, k, tau, r, q, sig, opt_type) + alpha * s_optimum * ((s / s_optimum) ** lam)
        else:
            price_ame = eta * (s - k)
        return price_ame

    @staticmethod
    def imp_vol(s: float, k: float, tau: float, r: float, q: float, price: float, opt_type: OptionType):
        vol_lb = 1e-4
        vol_ub = 10.0

        class PriceFunction(IUnivariateFunction):
            def evaluate(self, x):
                return BAW.price(s, k, tau, r, q, x, opt_type) - price

        pf = PriceFunction()
        bt = Brent(pf, vol_lb, vol_ub)
        vol = bt.solve()
        return vol

    @staticmethod
    def delta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_optimum = BAW.calc_s_optimum(k, tau, r, q, sig, opt_type)
        if eta * s < eta * s_optimum:
            delta_eur = BSM.delta(s, k, tau, r, q, sig, opt_type)
            delta_opt = BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)
            lam = BAW.calc_lambda(tau, r, q, sig, opt_type)
            return delta_eur + (eta - delta_opt) * ((s / s_optimum) ** (lam - 1.0))
        else:
            return eta

    @staticmethod
    def gamma(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_optimum = BAW.calc_s_optimum(k, tau, r, q, sig, opt_type)
        if eta * s < eta * s_optimum:
            gamma_eur = BSM.gamma(s, k, tau, r, q, sig)
            delta_opt = BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)
            lam = BAW.calc_lambda(tau, r, q, sig, opt_type)
            return gamma_eur + (eta - delta_opt) * (lam - 1.0) * ((s / s_optimum) ** (lam - 2.0)) / s_optimum
        else:
            return 0.0

    @staticmethod
    def theta(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        price = BAW.price(s, k, tau, r, q, sig, opt_type)
        delta = BAW.delta(s, k, tau, r, q, sig, opt_type)
        gamma = BAW.gamma(s, k, tau, r, q, sig, opt_type)
        theta = r * price - (r - q) * s * delta - 0.5 * (sig ** 2) * (s ** 2) * gamma
        return theta
