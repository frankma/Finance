import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm

from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedgingValuator import SingleVariableDeltaHedgingValuator
from src.SolverMC.SingleVariableSimulator import SingleVariableSimulator

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
        # set up all required states
        self.s_curr = self.simulator.curr
        self.s_prev = self.s_curr.copy()
        self.opt_amount = np.full(simulator.n_scenarios, -1.0)  # sell one option on one share as always
        self.share_amount = valuator.delta(self.s_curr, tau)  # initial delta
        self.cash_acc = valuator.price(self.s_curr, tau)  # premium income upon set up
        self.cash_acc -= self.s_curr * self.share_amount  # self-financing
        self.reb_count = np.ones(simulator.n_scenarios)  # rebalance counter

    def evolve(self, dt: float):
        self.cash_acc *= np.exp(dt * self.rf)  # cash account accumulate interest rate
        self.tau -= dt
        if self.tau < 0.0:
            raise AttributeError('option is already expired after incremental time %r.' % dt)
        self.s_prev = self.s_curr.copy()
        self.s_curr = self.simulator.evolve(dt)
        pass

    def rebalance(self):
        # indicator to apply rebalance
        reb = np.abs(self.s_curr / self.s_prev - 1.0) > self.threshold
        delta = self.valuator.delta(self.s_curr, self.tau)
        reb_value = (self.share_amount - delta) * self.s_curr  # rebalance cash amount
        self.share_amount[reb] = delta[reb]
        self.cash_acc[reb] += reb_value[reb]
        self.reb_count[reb] += 1
        pass

    def evaluate(self):
        option_acc = self.opt_amount * self.valuator.price(self.s_curr, self.tau)
        stock_acc = self.share_amount * self.s_curr
        return self.cash_acc + option_acc + stock_acc

    def graphical_analysis(self):
        value = self.evaluate()
        mean = np.average(value)
        std = np.std(value)
        bins = np.linspace(-5.0, 5.0, num=101)
        norms = norm.pdf(bins, mean, std)
        plt.hist(value, bins, normed=True)
        plt.plot(bins, norms)
        plt.title('hedged value at ttm of %r' % self.tau)
        plt.xlim([bins[0], bins[-1]])
        plt.ylim([0.0, 1.0])
        plt.show()

    def simulate_to_terminal(self):
        for idx, ttm in enumerate(self.taus[1:]):
            dt = self.taus[idx] - ttm
            self.evolve(dt)
            self.rebalance()
        pass
