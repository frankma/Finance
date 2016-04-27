import logging
import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class RdmBivariate(object):
    @staticmethod
    def draw(rho: float, size: int):
        x1 = np.random.random(size=size)
        x2 = np.random.random(size=size)
        return rho * x1 + np.sqrt(1.0 - rho * rho) * x2

    @staticmethod
    def pdf(x_1: np.array, x_2: np.array, rho: float, mu_1: float = 0.0, mu_2: float = 0.0,
            sig_1: float = 1.0, sig_2: float = 1.0):
        if abs(rho) >= 1.0:
            raise ValueError('rho (%.4f) should be smaller than 1' % rho)
        x_1_norm = (x_1 - mu_1) / sig_1
        x_2_norm = (x_2 - mu_2) / sig_2
        z = (x_1_norm ** 2) - 2.0 * rho * x_1_norm * x_2_norm + (x_2_norm ** 2)
        return np.exp(-z / (2.0 * np.sqrt(1.0 - rho ** 2))) / (2.0 * np.pi * sig_1 * sig_2 * np.sqrt(1.0 - rho ** 2))
