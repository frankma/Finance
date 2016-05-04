import logging

import numpy as np

from src.Utils.Types.VolType import VolType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SABRModel(object):
    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho
        self.abs_tol = 1e-12

    def sim_fwd_den(self, forward: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500, n_steps: int = 100,
                    n_scenarios: int = 10 ** 6):
        taus = np.linspace(self.t, 0.0, num=n_steps)
        bins = np.linspace(rel_bounds[0] * forward, rel_bounds[1] * forward, num=n_bins)
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
        freq, bins = np.histogram(forwards, bins=bins, normed=True)
        bins_mid = 0.5 * (bins[:-1] + bins[1:])
        return freq, bins_mid

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
