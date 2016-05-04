import logging

import numpy as np
from numpy.polynomial.polynomial import polyroots

from src.SABRModel.SABRModel import SABRModel
from src.Utils.Types.OptionType import OptionType
from src.Utils.Types.VolType import VolType
from src.Utils.Valuator.NormalModel import NormalModelVec

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


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
            ln_f_per_k = np.log(forward / strike)
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
            raise NotImplementedError('method not developed yet')
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
        # since normal model analytical greeks is incorrect as a collection of normal distribution with variable vols
        prices = NormalModelVec.price(forward, strikes, self.t, vols, 1.0, OptionType.put)
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
