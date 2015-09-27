import numpy as np

from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedgingValuator import SingleVariableDeltaHedgingValuator
from src.Simulator.SingleVariableSimulator import SingleVariableSimulator

__author__ = 'frank.ma'


class SingleVariableDeltaHedging(object):

    def __init__(self, simulator: SingleVariableSimulator, valuator: SingleVariableDeltaHedgingValuator,
                 n_steps: int, tau: float, rf: float, threshold: float):
        self.simulator = simulator
        self.valuator = valuator
        self.n_steps = n_steps
        self.tau = tau
        self.rf = rf
        self.threshold = threshold
        self.taus = np.linspace(tau, 0.0, num=n_steps + 1)
        # set underlying states
        self.s_curr = self.simulator.curr
        self.s_prev = self.s_curr.copy()
        self.opt_amount = np.full(simulator.n_scenarios, -1.0)  # sell one option on one share as always
        self.share_amount = valuator.delta(self.s_curr, tau)  # initial delta
        self.cash_acc = valuator.price(self.s_curr, tau)  # premium income
        self.cash_acc -= self.s_curr * self.share_amount  # self-financing
        self.reb_count = np.ones(simulator.n_scenarios)  # rebalancing counter

    def evolving(self, dt: float):
        self.cash_acc *= np.full(self.simulator.n_scenarios, np.exp(dt * self.rf))  # cash account accumulate interest rate
        self.tau -= dt
        if self.tau < 0.0:
            raise AttributeError('option is already expired after incremental time %r.' % dt)
        self.s_prev = self.s_curr.copy()
        self.s_curr = self.simulator.evolve(dt)
        pass

    def rebalancing(self):
        # indicator to
        reb = np.abs(self.s_curr / self.s_prev - np.ones(self.simulator.n_scenarios)) \
              > np.full(self.simulator.n_scenarios, self.threshold)
        delta = self.valuator.delta(self.s_curr, self.tau)
        self.share_amount[reb] = delta[reb]  # only update
        reb_value = (self.share_amount - delta) * self.s_curr  # assuming all scenarios rebalance
        self.cash_acc[reb] += reb_value[reb]
        self.reb_count[reb] += 1
        pass

    def simulating_to_terminal(self):
        for idx, ttm in enumerate(self.taus[1:]):
            dt = self.taus[idx] - ttm
            self.evolving(dt)
            self.rebalancing()
        pass
