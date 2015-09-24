import numpy as np
from scipy.stats import norm

__author__ = 'frank.ma'


class SingleVariableSimulator(object):

    def __init__(self, n_scenarios: int, init: float, drift: float, vol: float, model: str):

        self.n_scenarios = n_scenarios
        self.init = init
        self.drift = drift
        self.vol = vol
        if model.lower() == 'lognormal':
            self.model = 'lognormal'
        elif model.lower() == 'normal':
            self.model = 'normal'
        else:
            raise AttributeError('Unrecognized model %s, expect either Normal or LogNormal.')
        self.curr = np.full(n_scenarios, init)
        self.t = 0.0

    def evolve(self, dt: float):

        if dt < 0.0:
            raise AttributeError('Time incremental %r should be strictly positive.' % dt)

        self.t += dt

        rand = norm.pdf(np.random.random(self.n_scenarios))
        if self.model == 'lognormal':
            self.curr *= np.exp((self.drift - 0.5 * self.vol**2) * dt + self.vol * rand * np.sqrt(dt))
        else:
            self.curr += self.drift * dt + self.vol * rand * np.sqrt(dt)
