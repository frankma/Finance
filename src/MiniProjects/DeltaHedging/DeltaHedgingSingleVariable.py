from math import exp

import numpy as np

from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedgingValuator import SingleVariableDeltaHedgingValuator
from src.Simulator.SingleVariableSimulator import SingleVariableSimulator
from src.Utils import OptionType
from src.Utils.BSM import BSM, BSMVecS

__author__ = 'frank.ma'


class DeltaHedgingSingleVariable(object):

    def __init__(self, simulator: SingleVariableSimulator, valuator: SingleVariableDeltaHedgingValuator,
                 n_steps: int, tau: float, rf: float):
        self.simulator = simulator
        self.valuator = valuator
        self.n_steps = n_steps
        self.tau = tau
        self.rf = rf
        self.taus = np.linspace(tau, 0.0, num=n_steps + 1)
        # set underlying states
        self.s_curr = self.simulator.curr
        self.s_prev = self.s_curr.copy()
        self.opt_amount = np.full(simulator.n_scenarios, -1.0)  # sell one option on one share as always
        self.share_amount = valuator.delta(self.s_curr, tau)  # initial delta
        self.cash_acc = valuator.price(self.s_curr, tau)  # premium income
        self.cash_acc -= self.s_curr * self.share_amount  # self-financing
        self.reb_count = np.ones(simulator.n_scenarios)  # rebalancing counter

    def marching(self, dt: float):
        pass
