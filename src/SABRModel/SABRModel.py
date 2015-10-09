from math import log, sqrt

import numpy as np
from src.Utils.Black76 import Black76VecK
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class SABRModel(object):

    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def calc_lognormal_vol(self, forward: float, strike: float):
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strike
        f_per_k = forward / strike

        term1, term2, term3 = 1.0, 1.0, 1.0
        if abs(strike - forward) < 1e-10:
            # At-The-Money option reduces term1 and term2 as log(f / k_atm) is zero
            term1 = self.alpha / forward**one_m_beta
        else:
            z = self.nu / self.alpha * f_mul_k**(one_m_beta / 2.0) * log(f_per_k)
            x = log((sqrt(1.0 - 2.0 * self.rho * z + z**2) + z - self.rho) / (1.0 - self.rho))
            term1 = self.alpha / (f_mul_k**(one_m_beta / 2.0) * (1.0 + one_m_beta**2 / 24.0 * (log(f_per_k))**2 +
                                                                 one_m_beta**4 / 1920.0 * (log(f_per_k))**4))
            term2 = z / x
        term3 = (1.0 + (one_m_beta**2 / 24.0 * self.alpha**2 / (f_mul_k**one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k**(one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho**2) / 24.0 * self.nu**2) * self.t)

        return term1 * term2 * term3

    def calc_lognormal_vol_vec_k(self, forward: float, strikes: np.array) -> np.array:
        is_not_atm = np.abs(strikes - forward) > 1e-6
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strikes
        f_per_k = forward / strikes

        term1, term2, term3 = np.ones(strikes.__len__()), np.ones(strikes.__len__()), np.ones(strikes.__len__())

        z = self.nu / self.alpha * f_mul_k**(one_m_beta / 2.0) * np.log(f_per_k)
        x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z**2) + z - self.rho) / (1.0 - self.rho))

        term1 = self.alpha / (f_mul_k**(one_m_beta / 2.0) * (1.0 + one_m_beta**2 / 24.0 * (np.log(f_per_k))**2 +
                                                             one_m_beta**4 / 1920.0 * (np.log(f_per_k))**4))
        term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
        term3 = (1.0 + (one_m_beta**2 / 24.0 * self.alpha**2 / (f_mul_k**one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k**(one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho**2) / 24.0 * self.nu**2) * self.t)

        return term1 * term2 * term3

    def calc_lognormal_vol_vec_f(self, forwards: np.array, strike: float) -> np.array:
        is_not_atm = np.abs(strike - forwards) > 1e-6
        one_m_beta = 1.0 - self.beta
        f_mul_k = forwards * strike
        f_per_k = forwards / strike

        term1, term2, term3 = np.ones(forwards.__len__()), np.ones(forwards.__len__()), np.ones(forwards.__len__())

        z = self.nu / self.alpha * f_mul_k**(one_m_beta / 2.0) * np.log(f_per_k)
        x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z**2) + z - self.rho) / (1.0 - self.rho))

        term1 = self.alpha / (f_mul_k**(one_m_beta / 2.0) * (1.0 + one_m_beta**2 / 24.0 * (np.log(f_per_k))**2 +
                                                             one_m_beta**4 / 1920.0 * (np.log(f_per_k))**4))
        term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
        term3 = (1.0 + (one_m_beta**2 / 24.0 * self.alpha**2 / (f_mul_k**one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k**(one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho**2) / 24.0 * self.nu**2) * self.t)

        return term1 * term2 * term3

    def sim_forward_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500, n_steps: int = 100,
                        n_scenarios: int = 10**6):
        taus = np.linspace(self.t, 0.0, num=n_steps)
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins)
        # 1st, simulate forwards
        forwards = np.full(n_scenarios, forward)
        sigmas = np.full(n_scenarios, self.alpha)
        mean = [0.0, 0.0]
        correlation = [[1.0, self.rho], [self.rho, 1.0]]
        for idx, tau in enumerate(taus[1:]):
            dt = taus[idx] - tau
            sqrt_dt = np.sqrt(dt)
            rands = np.random.multivariate_normal(mean, correlation, size=n_scenarios)
            forwards *= np.exp(-0.5 * sigmas**2 * dt + sigmas * rands[:, 0] * sqrt_dt)
            sigmas *= np.exp(-0.5 * self.nu**2 * dt + self.nu * rands[:, 1] * sqrt_dt)
        # 2nd, analyse the density
        freq, strikes = np.histogram(forwards, bins=strikes, normed=True)
        strikes_mid = 0.5 * (strikes[:-1] + strikes[1:])
        return freq, strikes_mid

    def calc_forward_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins)
        vols = self.calc_lognormal_vol_vec_k(forward, strikes)
        # density = Black76VecK.gamma_k(forward, strikes, self.t, vols, 1.0)  # bond is zero as it is under fwd measure
        prices = Black76VecK.price(forward, strikes, self.t, vols, 1.0, OptionType.call)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1])**2)
        return density, strikes[1:-1]

    def calc_normal_vol(self, f: float, k: float):
        pass
