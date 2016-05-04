import logging

import numpy as np
from numpy.polynomial.polynomial import polyroots

from src.SABRModel.SABRModel import SABRModel
from src.Utils.Types.OptionType import OptionType
from src.Utils.Types.VolType import VolType
from src.Utils.Valuator.Black76 import Black76Vec

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SABRModelLognormalApprox(SABRModel):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        super().__init__(t, alpha, beta, nu, rho)

    def calc_vol(self, forward: float, strike: float, vol_type: VolType = VolType.black) -> float:
        one_m_beta = 1.0 - self.beta
        f_min_k = forward - strike
        f_mul_k = forward * strike
        ln_f_per_k = np.log(forward / strike)

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
            raise NotImplementedError('method not developed yet')
        else:
            msg = 'unrecognized volatility type %s' % vol_type.value
            logger.error(msg)
            raise ValueError(msg)

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
            msg = 'only one vectorization is allowed for either forward (%i) or strike (%i)' % (n_forward, n_strike)
            logger.error(msg)
            raise ValueError(msg)
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
            raise NotImplementedError('method not developed yet')
        else:
            raise ValueError('unrecognized volatility type %s' % vol_type.value)

    def calc_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins + 2)
        vols = self.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        # must implied through numerical differentiation
        # since Black 76 analytical greek is incorrect as a collection of lognormal distribution with variable vols
        prices = Black76Vec.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
        density = (prices[:-2] + prices[2:] - 2 * prices[1:-1]) / ((strikes[2:] - strikes[1:-1]) ** 2)
        strikes = strikes[1:-1]  # truncate strikes for numerical solution
        return density, strikes

    def calc_fwd_den_sp(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500):
        # special case for forward density function
        strikes = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins)
        vols = self.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        b = 1.0  # forward valuation, discount must be zero
        gamma_k = Black76Vec.gamma_k(forward, strikes, self.t, vols, b)
        vanna_k = Black76Vec.vanna_k(forward, strikes, self.t, vols, b)
        vomma = Black76Vec.vomma(forward, strikes, self.t, vols, b)
        vega = Black76Vec.vega(forward, strikes, self.t, vols, b)
        d_black_d_k = self.calc_d_black_d_k(forward, strikes)
        d2_black_d_k2 = self.calc_d2_black_d_k2(forward, strikes)
        density = gamma_k + 2.0 * vanna_k * d_black_d_k + vomma * (d_black_d_k ** 2) + vega * d2_black_d_k2
        return density, strikes

    def calc_loc_vol_vec(self, forward: float, strikes: np.array, mu: float) -> np.array:
        if abs(self.beta - 1.0) > self.abs_tol:
            raise ValueError('current method only support reduced form in which beta %r must be one.' % self.beta)

        black = self.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        d_black_d_t = self.calc_d_black_d_t(forward, strikes)
        d_black_d_k = self.calc_d_black_d_k(forward, strikes)
        d2_black_d_k2 = self.calc_d2_black_d_k2(forward, strikes)

        num = ((black ** 2) + 2.0 * black * self.t * (d_black_d_t + mu * strikes * d_black_d_k))
        den = ((1.0 - d_black_d_k * strikes * np.log(strikes / forward) / black) ** 2) + strikes * black * self.t * (
            d_black_d_k - 0.25 * strikes * black * self.t * (d_black_d_k ** 2) + strikes * d2_black_d_k2)
        loc_variance = num / den

        return np.sqrt(loc_variance)

    def calc_d_black_d_t(self, forward: float, strikes: np.array) -> np.array:
        is_atm = np.abs(forward - strikes) < self.abs_tol
        is_not_atm = np.logical_not(is_atm)
        d_black_d_t = np.ones(np.shape(strikes))
        z = self._calc_z(forward, strikes)
        x = self._calc_x(z)
        # use expansion for atm ones which has overflow issue
        d_black_d_t[is_atm] = 1.0 - 0.5 * self.rho * z[is_atm] + (-(self.rho ** 2) / 4.0 + 1.0 / 6.0) * (z[is_atm] ** 2)
        d_black_d_t[is_not_atm] *= z[is_not_atm] / x[is_not_atm]
        d_black_d_t *= self.alpha * (self.__calc_const() - 1.0) / self.t  # reverse calculation to keep commonality
        return d_black_d_t

    def calc_d_black_d_f(self, forward: float, strikes: np.array) -> np.array:
        d_black_d_k = self.nu / forward * self.__calc_d_z_per_x_d_z(forward, strikes)
        d_black_d_k *= self.__calc_const()

        return d_black_d_k

    def calc_d2_black_d_f2(self, forward: float, strikes: np.array) -> np.array:
        forward_sq = forward ** 2
        d2_black_d_k2 = (-self.nu / forward_sq * self.__calc_d_z_per_x_d_z(forward, strikes) +
                         self.nu ** 2 / self.alpha / forward_sq * self.__calc_d2_z_per_x_d_z2(forward, strikes))
        d2_black_d_k2 *= self.__calc_const()

        return d2_black_d_k2

    def calc_d_black_d_k(self, forward: float, strikes: np.array) -> np.array:
        d_black_d_k = -self.nu * self.__calc_d_z_per_x_d_z(forward, strikes) / strikes
        d_black_d_k *= self.__calc_const()

        return d_black_d_k

    def calc_d2_black_d_k2(self, forward: float, strikes: np.array) -> np.array:
        strikes_sq = strikes ** 2
        d2_black_d_k2 = self.nu * self.__calc_d_z_per_x_d_z(forward, strikes) / strikes_sq + \
                        (self.nu ** 2) * self.__calc_d2_z_per_x_d_z2(forward, strikes) / self.alpha / strikes_sq
        d2_black_d_k2 *= self.__calc_const()

        return d2_black_d_k2

    def __calc_d_z_per_x_d_z(self, forward: float, strikes: np.array) -> np.array:
        # calculate first order partial derivative (d (z/(x(z))))/(d z)
        is_not_atm = np.abs(forward - strikes) > 1e-12
        is_atm = np.logical_not(is_not_atm)
        dg_dz = np.ones(np.shape(strikes))  # g denotes the function of z / x(z)

        z = self._calc_z(forward, strikes)
        x = self._calc_x(z)
        s = np.sqrt(1.0 - 2.0 * self.rho * z + (z ** 2))  # s denotes 1 / (dx / dz)
        rho_sq = self.rho ** 2
        dg_dz[is_atm] = -self.rho / 2.0 + 2.0 * (-rho_sq / 4.0 + 1.0 / 6.0) * z[is_atm] - \
                        (6.0 * rho_sq - 5.0) * self.rho * (z[is_atm] ** 2) / 8.0
        dg_dz[is_not_atm] = (x[is_not_atm] * s[is_not_atm] - z[is_not_atm]) / (x[is_not_atm] ** 2) / s[is_not_atm]

        return dg_dz

    def __calc_d2_z_per_x_d_z2(self, forward: float, strikes: np.array) -> np.array:
        # calculate second order partial derivative (d^2 (z/x(z)))/(d z^2)
        is_not_atm = np.abs(forward - strikes) > 1e-12
        is_atm = np.logical_not(is_not_atm)
        d2g_dz2 = np.ones(np.shape(strikes))  # g denotes the function of z / x(z)

        z = self._calc_z(forward, strikes)
        x = self._calc_x(z)
        s = np.sqrt(1.0 - 2.0 * self.rho * z + (z ** 2))  # s denotes 1 / (dx / dz)
        rho_sq = self.rho ** 2
        d2g_dz2[is_atm] = 2.0 * (-rho_sq / 4.0 + 1.0 / 6.0) - (6.0 * rho_sq - 5.0) * self.rho * z[is_atm] / 2.0 + \
                          12.0 * (-5.0 * (rho_sq ** 2) / 16.0 + rho_sq / 3.0 - 17.0 / 360.0) * (z[is_atm] ** 2)
        d2g_dz2[is_not_atm] = (x[is_not_atm] * (3.0 * self.rho * z[is_not_atm] - (z[is_not_atm] ** 2) - 2.0) +
                               2.0 * s[is_not_atm] * z[is_not_atm]) / ((x[is_not_atm] * s[is_not_atm]) ** 3)

        return d2g_dz2

    def __calc_const(self):
        const = (1.0 + (self.alpha * self.nu * self.rho / 4.0 +
                        (2.0 - 3.0 * (self.rho ** 2)) * (self.nu ** 2) / 24.0) * self.t)
        return const

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
