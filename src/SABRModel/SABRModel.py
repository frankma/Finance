from math import log, sqrt

import numpy as np
from numpy.polynomial.polynomial import polyroots

from src.Utils.VolType import VolType
from src.Utils.Black76 import Black76VecK
from src.Utils.NormalModel import NormalModelVecK
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class SABRModel(object):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def sim_forward_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500, n_steps: int = 100,
                        n_scenarios: int = 10 ** 6):
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
            forwards += sigmas * (forwards ** self.beta) * rands[:, 0] * sqrt_dt
            # use lognormal transform to avoid negative volatility
            sigmas *= np.exp(-0.5 * (self.nu ** 2) * dt + self.nu * rands[:, 1] * sqrt_dt)
        # 2nd, analyse the density
        freq, strikes = np.histogram(forwards, bins=strikes, normed=True)
        strikes_mid = 0.5 * (strikes[:-1] + strikes[1:])
        return freq, strikes_mid

    def calc_vol(self, forward: float, strike: float, vol_type: VolType) -> float:
        pass

    def calc_vol_vec_k(self, forward: float, strikes: np.array, vol_type: VolType) -> np.array:
        pass

    def calc_vol_vec_f(self, forwards: np.array, strike: float, vol_type: VolType) -> np.array:
        pass

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        pass

    @staticmethod
    def solve_alpha(forward: float, vol_atm: float, t: float, beta: float, nu: float, rho: float,
                    vol_type: VolType = VolType.black) -> float:
        f_pwr_one_m_beta = forward ** (1.0 - beta)

        if vol_type == VolType.black:
            cubic = ((1.0 - beta) ** 2) * t / 24.0 / (f_pwr_one_m_beta ** 3)
            quadratic = beta * nu * rho * t / 4.0 / (f_pwr_one_m_beta ** 2)
            linear = (1.0 + (nu ** 2) * (2.0 - 3.0 * (rho ** 2)) * t / 24.0) / f_pwr_one_m_beta
            constant = -vol_atm

            alpha_approximate = vol_atm / f_pwr_one_m_beta  # in case of multiple real solutions
        elif vol_type == VolType.normal:
            cubic = -beta * (2.0 - beta) * t / 24.0 / (forward ** (2.0 - 3.0 * beta))
            quadratic = beta * nu * rho * t / 4.0 / (forward ** (1.0 - 2.0 * beta))
            linear = (1.0 + (nu ** 2) * (2.0 - 3.0 * (rho ** 2)) * t / 24.0) * (forward ** beta)
            constant = -vol_atm

            alpha_approximate = vol_atm / (forward ** beta)  # in case of multiple real solutions
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

        coefficients = np.array([constant, linear, quadratic, cubic])
        solutions = polyroots(coefficients)
        solutions = solutions[np.isreal(solutions)].real  # only real solution is usable
        n_solutions = solutions.__len__()
        if n_solutions < 1 or all(solutions < 0.0):
            raise ValueError('cannot find alpha within real domain')
        elif n_solutions == 1:
            return solutions[0]
        else:
            closest = min(solutions, key=lambda x: abs(x - alpha_approximate))
            return closest


class SABRModelLognormalApprox(SABRModel):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        super().__init__(t, alpha, beta, nu, rho)

    def calc_vol(self, forward: float, strike: float, vol_type: VolType = VolType.black) -> float:
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strike
        f_per_k = forward / strike

        term1, term2, term3 = 1.0, 1.0, 1.0
        if abs(strike - forward) < 1e-10:
            # At-The-Money option reduces term1 and term2 as log(f / k_atm) is zero
            term1 = self.alpha / forward ** one_m_beta
        else:
            z = self.nu / self.alpha * f_mul_k ** (one_m_beta / 2.0) * log(f_per_k)
            x = log((sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term1 = self.alpha / (f_mul_k ** (one_m_beta / 2.0) *
                                  (1.0 + one_m_beta ** 2 / 24.0 *
                                   (log(f_per_k)) ** 2 + one_m_beta ** 4 / 1920.0 * (log(f_per_k)) ** 4))
            term2 = z / x
        term3 = (1.0 + (one_m_beta ** 2 / 24.0 * self.alpha ** 2 / (f_mul_k ** one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k ** (one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t)
        black_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            return black_vol
        elif vol_type == VolType.normal:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_vol_vec_k(self, forward: float, strikes: np.array, vol_type: VolType = VolType.black) -> np.array:
        is_not_atm = np.abs(strikes - forward) > 1e-6
        n = strikes.__len__()
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strikes
        f_per_k = forward / strikes

        term1, term2, term3 = np.ones(n), np.ones(n), np.ones(n)
        z = self.nu / self.alpha * f_mul_k ** (one_m_beta / 2.0) * np.log(f_per_k)
        x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
        term1 = self.alpha / (f_mul_k ** (one_m_beta / 2.0) * (1.0 + one_m_beta ** 2 / 24.0 * (np.log(f_per_k)) ** 2 +
                                                               one_m_beta ** 4 / 1920.0 * (np.log(f_per_k)) ** 4))
        term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
        term3 = (1.0 + (one_m_beta ** 2 / 24.0 * self.alpha ** 2 / (f_mul_k ** one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k ** (one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t)
        black_vols = term1 * term2 * term3

        if vol_type == VolType.black:
            return black_vols
        elif vol_type == VolType.normal:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_vol_vec_f(self, forwards: np.array, strike: float, vol_type: VolType = VolType.black) -> np.array:
        is_not_atm = np.abs(strike - forwards) > 1e-6
        one_m_beta = 1.0 - self.beta
        f_mul_k = forwards * strike
        f_per_k = forwards / strike

        term1, term2, term3 = np.ones(forwards.__len__()), np.ones(forwards.__len__()), np.ones(forwards.__len__())

        z = self.nu / self.alpha * f_mul_k ** (one_m_beta / 2.0) * np.log(f_per_k)
        x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))

        term1 = self.alpha / (f_mul_k ** (one_m_beta / 2.0) * (1.0 + one_m_beta ** 2 / 24.0 * (np.log(f_per_k)) ** 2 +
                                                               one_m_beta ** 4 / 1920.0 * (np.log(f_per_k)) ** 4))
        term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
        term3 = (1.0 + (one_m_beta ** 2 / 24.0 * self.alpha ** 2 / (f_mul_k ** one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k ** (one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t)
        black_vols = term1 * term2 * term3

        if vol_type == VolType.black:
            return black_vols
        elif vol_type == VolType.normal:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins + 2)
        vols = self.calc_vol_vec_k(forward, strikes, vol_type=VolType.black)
        # must implied through numerical differentiation
        # since analytical one is incorrect as a collection of lognormal distribution with variable vols
        prices = Black76VecK.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1]) ** 2)
        strikes = strikes[1:-1]  # truncate strikes for numerical solution
        return density, strikes


class SABRModelNormalApprox(SABRModel):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        super().__init__(t, alpha, beta, nu, rho)

    def calc_vol(self, forward: float, strike: float, vol_type: VolType = VolType.normal) -> float:
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strike

        term1, term2, term3 = self.alpha * (f_mul_k ** (self.beta / 2.0)), 1.0, 1.0
        if abs(forward - strike) > 1e-10:
            if abs(self.beta) < 1e-10:
                z = self.nu / self.alpha * (forward - strike)
                term3 = 1.0 + (2.0 - 3.0 * self.rho ** 2) / 24.0 * (self.nu ** 2) * self.t
            else:
                f_per_k = forward / strike
                ln_f_per_k = log(f_per_k)
                term1 *= (1.0 + (ln_f_per_k ** 2) / 24.0 + (ln_f_per_k ** 4) / 1920.0) / \
                         (1.0 + (one_m_beta ** 2) / 24.0 * (ln_f_per_k ** 2) + (one_m_beta ** 4) / 1920.0 * (
                             ln_f_per_k ** 4))
                z = self.nu / self.alpha * f_mul_k ** (one_m_beta / 2.0) * log(f_per_k)
                term3 = 1.0 + (-self.beta * (2.0 - self.beta) * self.alpha ** 2 / 24.0 / (f_mul_k ** one_m_beta) +
                               self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                               (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
            x = log((sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term2 = z / x
        else:
            if abs(self.beta) < 1e-10:
                term3 = 1.0 + ((2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
            else:
                term3 = 1.0 + (-self.beta * (2.0 - self.beta) * self.alpha ** 2 / 24.0 / (f_mul_k ** one_m_beta) +
                               self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                               (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        normal_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        elif vol_type == VolType.normal:
            return normal_vol
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_vol_vec_k(self, forward: float, strikes: np.array, vol_type: VolType = VolType.normal) -> np.array:
        is_not_atm_or_zero_k = np.logical_and(np.abs(forward - strikes) > 1e-12, np.abs(strikes) > 1e-12)
        n = strikes.__len__()
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strikes

        term1, term2, term3 = self.alpha * (f_mul_k ** (self.beta / 2.0)), np.ones(n), np.ones(n)
        if abs(self.beta) < 1e-12:
            z = self.nu / self.alpha * (forward - strikes)
            x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term2[is_not_atm_or_zero_k] = z[is_not_atm_or_zero_k] / x[is_not_atm_or_zero_k]
            term3 = 1.0 + ((2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        else:
            f_per_k = forward / strikes
            ln_f_per_k = np.log(f_per_k)
            term1 *= (1.0 + (ln_f_per_k ** 2) / 24.0 + (ln_f_per_k ** 4) / 1920.0) / \
                     (1.0 + (one_m_beta ** 2) / 24.0 * (ln_f_per_k ** 2) + (one_m_beta ** 4) / 1920.0 * (
                         ln_f_per_k ** 4))
            z = self.nu / self.alpha * (f_mul_k ** (one_m_beta / 2.0)) * np.log(f_per_k)
            x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term2[is_not_atm_or_zero_k] = z[is_not_atm_or_zero_k] / x[is_not_atm_or_zero_k]
            term3 = 1.0 + (-self.beta * (2.0 - self.beta) * self.alpha ** 2 / 24.0 / (f_mul_k ** one_m_beta) +
                           self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                           (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        normal_vols = term1 * term2 * term3

        if vol_type == VolType.black:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        elif vol_type == VolType.normal:
            return normal_vols
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_vol_vec_f(self, forwards: np.array, strike: float, vol_type: VolType = VolType.normal) -> np.array:
        is_not_atm_or_zero_f = np.logical_and(np.abs(forwards - strike) > 1e-12, np.abs(forwards) > 1e-12)
        n = forwards.__len__()
        one_m_beta = 1.0 - self.beta
        f_mul_k = forwards * strike

        term1, term2, term3 = self.alpha * (f_mul_k ** (self.beta / 2.0)), np.ones(n), np.ones(n)
        if abs(self.beta) < 1e-12:
            z = self.nu / self.alpha * (forwards - strike)
            x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term2[is_not_atm_or_zero_f] = z[is_not_atm_or_zero_f] / x[is_not_atm_or_zero_f]
            term3 = 1.0 + ((2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        else:
            f_per_k = forwards / strike
            ln_f_per_k = np.log(f_per_k)
            term1 *= (1.0 + (ln_f_per_k ** 2) / 24.0 + (ln_f_per_k ** 4) / 1920.0) / \
                     (1.0 + (one_m_beta ** 2) / 24.0 * (ln_f_per_k ** 2) + (one_m_beta ** 4) / 1920.0 * (
                         ln_f_per_k ** 4))
            z = self.nu / self.alpha * f_mul_k ** (one_m_beta / 2.0) * np.log(f_per_k)
            x = np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))
            term2[is_not_atm_or_zero_f] = z[is_not_atm_or_zero_f] / x[is_not_atm_or_zero_f]
            term3 = 1.0 + (-self.beta * (2.0 - self.beta) * self.alpha ** 2 / 24.0 / (f_mul_k ** one_m_beta) +
                           self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                           (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        normal_vols = term1 * term2 * term3

        if vol_type == VolType.black:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        elif vol_type == VolType.normal:
            return normal_vols
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.__str__())

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins + 2)
        vols = self.calc_vol_vec_k(forward, strikes, vol_type=VolType.normal)
        # must implied through numerical differentiation
        # since analytical one is incorrect as a collection of lognormal distribution with variable vols
        prices = NormalModelVecK.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1]) ** 2)
        strikes = strikes[1:-1]  # truncate strikes for numerical solution
        return density, strikes
