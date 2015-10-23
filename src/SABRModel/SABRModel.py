from math import log

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
        self.abs_tol = 1e-12

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

    def get_model_lognormal_approx(self):
        return SABRModelLognormalApprox(self.t, self.alpha, self.beta, self.nu, self.rho)

    def get_model_normal_approx(self):
        return SABRModelNormalApprox(self.t, self.alpha, self.beta, self.nu, self.rho)

    def _calc_z(self, forward, strike):
        return self.nu / self.alpha * np.log(forward / strike) * ((forward * strike) ** ((1.0 - self.beta) / 2.0))

    def _calc_z_norm(self, forward, strike):
        return self.nu / self.alpha * (forward - strike)

    def _calc_x(self, z):
        return np.log((np.sqrt(1.0 - 2.0 * self.rho * z + z ** 2) + z - self.rho) / (1.0 - self.rho))

    def calc_vol(self, forward: float, strike: float, vol_type: VolType) -> float:
        raise NotImplementedError('unexpected call of abstract method')

    def calc_vol_vec(self, forward: float or np.array, strikes: np.array or float, vol_type: VolType) -> np.array:
        raise NotImplementedError('unexpected call of abstract method')

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        raise NotImplementedError('unexpected call of abstract method')

    @staticmethod
    def solve_alpha(forward: float, vol_atm: float, t: float, beta: float, nu: float, rho: float) -> float:
        raise NotImplementedError('unexpected call of abstract method')


class SABRModelLognormalApprox(SABRModel):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        super().__init__(t, alpha, beta, nu, rho)

    def calc_vol(self, forward: float, strike: float, vol_type: VolType = VolType.black) -> float:
        one_m_beta = 1.0 - self.beta
        f_min_k = forward - strike
        f_mul_k = forward * strike
        ln_f_per_k = log(forward / strike)

        term1, term2, term3 = 1.0, 1.0, 1.0
        if abs(self.beta) < self.abs_tol:
            if abs(f_min_k) < self.abs_tol:
                term1 = self.alpha / forward
            else:
                term1 = self.alpha * ln_f_per_k / f_min_k
            term3 = 1.0 + ((self.alpha ** 2) / 24.0 / f_mul_k +
                           (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) / 24.0) * self.t
        else:
            term1 = self.alpha / (f_mul_k ** (one_m_beta / 2.0)) / \
                    (1.0 + (one_m_beta ** 2) * (ln_f_per_k ** 2) / 24.0 +
                     (one_m_beta ** 4) * (ln_f_per_k ** 4) / 1920.0)
            term3 = (1.0 + (one_m_beta ** 2 / 24.0 * self.alpha ** 2 / (f_mul_k ** one_m_beta) +
                            0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k ** (one_m_beta / 2.0) +
                            (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t)
        if abs(f_min_k) >= self.abs_tol:
            z = self._calc_z(forward, strike)
            x = self._calc_x(z)
            term2 = z / x
        black_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            return black_vol
        elif vol_type == VolType.normal:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.value)

    def calc_vol_vec(self, forward: float or np.array, strike: np.array or float,
                     vol_type: VolType = VolType.black) -> np.array:
        n_forward = np.size(forward)
        n_strike = np.size(strike)
        if n_forward == 1:
            n = np.size(strike)
            forward = np.full(n, forward)
        elif n_strike == 1:
            n = np.size(forward)
            strike = np.full(n, strike)
        else:
            raise ValueError('only one vectorization is allowed for either forward (%i) or strike (%i)'
                             % (n_forward, n_strike))
        f_min_k = forward - strike
        f_mul_k = forward * strike
        ln_f_per_k = np.log(forward / strike)
        is_atm = np.abs(f_min_k) < self.abs_tol
        is_not_atm = np.logical_not(is_atm)
        one_m_beta = 1.0 - self.beta

        term1, term2, term3 = np.ones(n), np.ones(n), np.ones(n)
        if abs(self.beta) < self.abs_tol:
            term1[is_not_atm] = self.alpha * ln_f_per_k[is_not_atm] / f_min_k[is_not_atm]
            term1[is_atm] = self.alpha / forward[is_atm]
            term3 = 1.0 + ((self.alpha ** 2) / 24.0 / f_mul_k +
                           (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) / 24.0) * self.t
        else:
            term1 = self.alpha / (f_mul_k ** (one_m_beta / 2.0)) / \
                    (1.0 + (one_m_beta ** 2) * (ln_f_per_k ** 2) / 24.0 +
                     (one_m_beta ** 4) * (ln_f_per_k ** 4) / 1920.0)
            term3 = (1.0 + (one_m_beta ** 2 / 24.0 * self.alpha ** 2 / (f_mul_k ** one_m_beta) +
                            0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k ** (one_m_beta / 2.0) +
                            (2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t)
        z = self._calc_z(forward, strike)
        x = self._calc_x(z)
        term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
        black_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            return black_vol
        elif vol_type == VolType.normal:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.value)

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins + 2)
        vols = self.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        # must implied through numerical differentiation
        # since analytical one is incorrect as a collection of lognormal distribution with variable vols
        prices = Black76VecK.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1]) ** 2)
        strikes = strikes[1:-1]  # truncate strikes for numerical solution
        return density, strikes

    @staticmethod
    def solve_alpha(forward: float, vol_atm: float, t: float, beta: float, nu: float, rho: float):
        f_pwr_one_m_beta = forward ** (1.0 - beta)
        cubic = ((1.0 - beta) ** 2) * t / 24.0 / (f_pwr_one_m_beta ** 3)
        quadratic = beta * nu * rho * t / 4.0 / (f_pwr_one_m_beta ** 2)
        linear = (1.0 + (nu ** 2) * (2.0 - 3.0 * (rho ** 2)) * t / 24.0) / f_pwr_one_m_beta
        constant = -vol_atm

        coefficients = np.array([constant, linear, quadratic, cubic])
        solutions = polyroots(coefficients)
        solutions = solutions[np.isreal(solutions)].real  # only real solution is usable
        n_solutions = solutions.__len__()
        if n_solutions < 1 or all(solutions < 0.0):
            raise ValueError('cannot find alpha within real domain')
        elif n_solutions == 1:
            return solutions[0]
        else:
            alpha_approximate = vol_atm / f_pwr_one_m_beta
            closest = min(solutions, key=lambda x: abs(x - alpha_approximate))
            return closest


class SABRModelNormalApprox(SABRModel):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        super().__init__(t, alpha, beta, nu, rho)

    def calc_vol(self, forward: float, strike: float, vol_type: VolType = VolType.normal) -> float:
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strike

        term1, term2, term3 = self.alpha, 1.0, 1.0
        if abs(self.beta) < self.abs_tol:
            if abs(forward - strike) >= self.abs_tol:
                z = self._calc_z_norm(forward, strike)
                x = self._calc_x(z)
                term2 = z / x
            term3 = 1.0 + ((2.0 - 3.0 * self.rho ** 2) / 24.0 * self.nu ** 2) * self.t
        else:
            ln_f_per_k = log(forward / strike)
            term1 *= (f_mul_k ** (self.beta / 2.0)) * (1.0 + (ln_f_per_k ** 2) / 24.0 + (ln_f_per_k ** 4) / 1920.0) / \
                     (1.0 + (one_m_beta ** 2) * (ln_f_per_k ** 2) / 24.0 +
                      (one_m_beta ** 4) * (ln_f_per_k ** 4) / 1920.0)
            if abs(forward - strike) >= self.abs_tol:
                z = self._calc_z(forward, strike)
                x = self._calc_x(z)
                term2 = z / x
            term3 = 1.0 + (-self.beta * (2.0 - self.beta) * (self.alpha ** 2) / 24.0 / (f_mul_k ** one_m_beta) +
                           self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                           (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) / 24.0) * self.t
        normal_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            # TODO: develop this method -- SABR refactor
            raise ValueError('method not developed yet')
        elif vol_type == VolType.normal:
            return normal_vol
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.value)

    def calc_vol_vec(self, forward: float or np.array, strike: np.array or float,
                     vol_type: VolType = VolType.normal) -> np.array:
        n_forward = np.size(forward)
        n_strike = np.size(strike)
        if n_forward == 1:
            n = np.size(strike)
        elif n_strike == 1:
            n = np.size(forward)
        else:
            raise ValueError('only one vectorization is allowed for either forward (%i) or strike (%i)'
                             % (n_forward, n_strike))
        is_not_atm = np.abs(forward - strike) > self.abs_tol
        one_m_beta = 1.0 - self.beta
        f_mul_k = forward * strike

        term1, term2, term3 = np.full(n, self.alpha), np.ones(n), np.ones(n)
        if abs(self.beta) < self.abs_tol:
            z = self._calc_z_norm(forward, strike)
            x = self._calc_x(z)
            term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
            term3 = 1.0 + (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) * self.t / 24.0
        else:
            ln_f_per_k = np.log(forward / strike)
            term1 *= (f_mul_k ** (self.beta / 2.0)) * (1.0 + (ln_f_per_k ** 2) / 24.0 + (ln_f_per_k ** 4) / 1920.0) / \
                     (1.0 + (one_m_beta ** 2) * (ln_f_per_k ** 2) / 24.0 +
                      (one_m_beta ** 4) * (ln_f_per_k ** 4) / 1920.0)
            z = self._calc_z(forward, strike)
            x = self._calc_x(z)
            term2[is_not_atm] = z[is_not_atm] / x[is_not_atm]
            term3 = 1.0 + (-self.beta * (2.0 - self.beta) * (self.alpha ** 2) / 24.0 / (f_mul_k ** one_m_beta) +
                           self.rho * self.alpha * self.nu * self.beta / 4.0 / (f_mul_k ** (one_m_beta / 2.0)) +
                           (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) / 24.0) * self.t
        normal_vol = term1 * term2 * term3

        if vol_type == VolType.black:
            # TODO: develop this method -- SABR vol conversion
            raise NotImplementedError('TODO')
        elif vol_type == VolType.normal:
            return normal_vol
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.value)

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins + 2)
        vols = self.calc_vol_vec(forward, strikes, vol_type=VolType.normal)
        # must implied through numerical differentiation
        # since analytical one is incorrect as a collection of lognormal distribution with variable vols
        prices = NormalModelVecK.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1]) ** 2)
        strikes = strikes[1:-1]  # truncate strikes for numerical solution
        return density, strikes

    @staticmethod
    def solve_alpha(forward: float, vol_atm: float, t: float, beta: float, nu: float, rho: float):
        cubic = -beta * (2.0 - beta) * t / 24.0 / (forward ** (2.0 - 3.0 * beta))
        quadratic = beta * nu * rho * t / 4.0 / (forward ** (1.0 - 2.0 * beta))
        linear = (1.0 + (nu ** 2) * (2.0 - 3.0 * (rho ** 2)) * t / 24.0) * (forward ** beta)
        constant = -vol_atm

        coefficients = np.array([constant, linear, quadratic, cubic])
        solutions = polyroots(coefficients)
        solutions = solutions[np.isreal(solutions)].real  # only real solution is usable
        n_solutions = solutions.__len__()
        if n_solutions < 1 or all(solutions < 0.0):
            raise ValueError('cannot find alpha within real domain')
        elif n_solutions == 1:
            return solutions[0]
        else:
            alpha_approximate = vol_atm / (forward ** beta)
            closest = min(solutions, key=lambda x: abs(x - alpha_approximate))
            return closest
