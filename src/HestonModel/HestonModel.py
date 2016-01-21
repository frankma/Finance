import logging

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class HestonModel(object):
    def __init__(self, mu: float, v_0: float, kappa: float, theta: float, nu: float, rho: float):
        self.mu = mu
        self.v_0 = v_0
        self.kappa = kappa
        self.theta = theta
        self.nu = nu
        self.rho = rho

    def sim_forward_den(self, spot: float, t: float, rel_bounds: tuple = (0.01, 20.0), n_bins: int = 500,
                        n_steps: int = 100, n_scenarios: int = 10 ** 6):
        ts = np.linspace(0.0, t, num=n_steps)
        bins = np.linspace(rel_bounds[0] * spot, rel_bounds[1] * spot, num=n_bins)
        spots = np.full(n_scenarios, spot)
        variances = np.full(n_scenarios, self.v_0)
        mean = [0.0, 0.0]
        correlation = [[1.0, self.rho], [self.rho, 1.0]]
        # 1st, simulate spot
        for idx, t_step in enumerate(ts[1:]):
            dt = t_step - ts[idx]
            sqrt_dt = np.sqrt(dt)
            volatilities = np.sqrt(variances)
            rands = np.random.multivariate_normal(mean, correlation, size=n_scenarios)
            spots += self.mu * spots * dt + volatilities * rands[:, 0] * sqrt_dt
            variances += self.kappa * (self.theta - variances) * dt + self.nu * volatilities * rands[:, 1] * sqrt_dt

        # 2nd, analyse density
        freq, bins = np.histogram(spots, bins, normed=True)
        bins_mid = 0.5 * (bins[:-1] + bins[1:])
        return freq, bins_mid
