import numpy as np

from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BAW(object):
    @staticmethod
    def price(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_optimum = BAW.__calc_s_optimum(s, k, tau, r, q, sig, opt_type)
        if s * eta < s_optimum * eta:
            beta = BAW.__calc_beta(tau, r, q, sig, opt_type)
            alpha = (eta - BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)) * s_optimum / beta
            price_ame = BSM.price(s, k, tau, r, q, sig, opt_type) + alpha * ((s / s_optimum) ** beta)
        else:
            price_ame = eta * (s - k)

        return price_ame

    @staticmethod
    def __calc_beta(tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        sig_sq = sig ** 2
        m = 2.0 * r / sig_sq
        n = 2.0 * (r - q) / sig_sq
        n_min_one = n - 1.0
        g = 1.0 - np.exp(-r * tau)
        beta = 0.5 * (-n_min_one + eta * np.sqrt((n_min_one ** 2) + 4.0 * m / g))
        return beta

    @staticmethod
    def __calc_s_optimum(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        s_opt = s  # use s as initial guess
        lhs = s_opt
        rhs = 0.0
        eta = opt_type.value
        while abs(lhs - eta * rhs) / k > 1e-6:
            price = BSM.price(s_opt, k, tau, r, q, sig, opt_type)
            delta = BSM.delta(s_opt, k, tau, r, q, sig, opt_type)
            gamma = BSM.gamma(s_opt, k, tau, r, q, sig)
            beta = BAW.__calc_beta(tau, r, q, sig, opt_type)
            lhs = eta * (s_opt - k)
            rhs = eta * price + (1 - eta * delta) * s_opt / beta
            slope = delta * (1.0 - 1.0 / beta) + (1.0 - gamma * s_opt) / beta
            s_opt = (k + rhs - slope * s_opt) / (1.0 - slope)

        return s_opt
