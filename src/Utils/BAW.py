import numpy as np

from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BAW(object):
    @staticmethod
    def price(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_optimum = BAW.__calc_s_optimum(k, tau, r, q, sig, opt_type)
        if eta * s < eta * s_optimum:
            lam = BAW.__calc_lam(tau, r, q, sig, opt_type)
            alpha = (eta - BSM.delta(s_optimum, k, tau, r, q, sig, opt_type)) / lam / (s_optimum ** (lam - 1))
            price_ame = BSM.price(s, k, tau, r, q, sig, opt_type) + alpha * (s ** lam)
        else:
            price_ame = eta * (s - k)
        return price_ame

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
    def __calc_s_optimum(k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):
        eta = opt_type.value
        s_opt = k  # use s as initial guess
        f = s_opt  # an arbitrary larger than tolerance value to start
        while abs(f) > 1e-12:
            price = BSM.price(s_opt, k, tau, r, q, sig, opt_type)
            delta = BSM.delta(s_opt, k, tau, r, q, sig, opt_type)
            gamma = BSM.gamma(s_opt, k, tau, r, q, sig)
            lam = BAW.__calc_lam(tau, r, q, sig, opt_type)
            f = eta * (s_opt - k) - price + (eta - delta) * s_opt / lam
            f_prime = (eta - delta) * (1.0 - 1.0 / lam) + gamma * s_opt / lam
            s_opt -= f / f_prime

        return s_opt
