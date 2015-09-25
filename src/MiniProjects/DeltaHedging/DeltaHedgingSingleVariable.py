import numpy as np

from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedgingValuator import SingleVariableDeltaHedgingValuator
from src.Simulator.SingleVariableSimulator import SingleVariableSimulator
from src.Utils import OptionType
from src.Utils.BSM import BSM

__author__ = 'frank.ma'


class DeltaHedgingSingleVariable(object):

    def __init__(self, n_scenarios: int, n_steps: int, s_0: float, k: float, tau: float, r_opt: float, q_opt: float,
                 vol_opt: float, opt_type: OptionType, r_sim: float, q_sim: float, vol_sim: float, model='lognormal'):
        self.tau = tau
        self.n_steps = n_steps
        self.taus = np.linspace(tau, 0.0, num=n_steps + 1)
        self.simulator = SingleVariableSimulator(n_scenarios, s_0, r_sim - q_sim, vol_sim, model)
        self.valuator = SingleVariableDeltaHedgingValuator(k, r_opt, q_opt, vol_opt, opt_type)
        # set underlying states
        self.s_curr = self.simulator.curr
        self.s_prev = self.s_curr.copy()
        self.opt_acc = np.full(n_scenarios, -1.0)  # sell one option on one share as always
        self.shr_acc = np.full(n_scenarios, BSM.delta(s_0, k, tau, r_opt, q_opt, vol_opt, opt_type))  # initial delta
        self.cash_acc = np.full(n_scenarios, BSM.price(s_0, k, tau, r_opt, q_opt, vol_opt, opt_type))  # premium income
        self.cash_acc -= self.s_curr * self.shr_acc  # self-financing

    def rebalancing(self):
        pass
