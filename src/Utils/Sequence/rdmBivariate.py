import numpy as np

__author__ = 'frank.ma'


class RdmBivariate(object):
    @staticmethod
    def draw(rho: float, size: int):
        x1 = np.random.random(size=size)
        x2 = np.random.random(size=size)
        return rho * x1 + np.sqrt(1.0 - rho * rho) * x2

    @staticmethod
    def cdf(x1: np.array, x2: np.array, rho: float):
        pass
