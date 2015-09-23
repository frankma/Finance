from math import sqrt

import numpy as np
from scipy.stats import norm

__author__ = 'frank.ma'


class SingleVarSimulation(object):

    def __init__(self, n_scn: int, s_0: float, taus: list, r: float, q: float, sig: float, model='lognormal'):
        self.n_scn = n_scn
        self.taus = taus
        self.s_0 = s_0
        self.r = r
        self.q = q
        self.sig = sig
        self.model = model

        self.s_curr = np.ones(self.n_scn) * self.s_0
        self.s_prev = self.s_curr.copy()

    def marching(self, dt: float):
        self.s_prev = self.s_curr.copy()
        self.s_curr = self.evolve(self.s_prev, dt, self.r, self.q, self.sig, self.model)
        return self.s_curr

    @staticmethod
    def evolve(s_prev: np.array, dt: float, r: float, q: float, sig: float, model='lognormal'):

        rand_norm = norm.ppf(np.random.random(s_prev.__len__()))

        if model == 'lognormal':
            return s_prev * np.exp((r - q - 0.5 * sig**2) * dt + sig * rand_norm * sqrt(dt))
        elif model == 'normal':
            return s_prev * (1.0 + (r - q) * dt + sig * rand_norm * sqrt(dt))
        else:
            raise ValueError('Unrecognized model input: %s' % model)

