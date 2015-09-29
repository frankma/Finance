from math import exp

import numpy as np

from src.MiniProjects.DeltaHedging.SingleVarSimulation import SingleVarSimulation
from src.Utils.BSM import BSMVecS
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class SingleVarDeltaHedging(object):

    FIX_COMMISSION = 1.0
    VAR_COMMISSION = 0.25
    REB_LEVEL = 0.01

    def __init__(self, n_scn: int, n_stp: float, s_0: float, k: float, tau: float, r_sim: float, q_sim: float,
                 sig_sim: float, r_opt: float, q_opt: float, sig_opt: float, opt_type: OptionType, model='lognormal'):
        self.n_scn = n_scn
        self.s_0 = s_0
        self.k = k
        self.taus = list(tau - np.array(range(0, n_stp + 1, 1)) * tau / n_stp)
        self.r_sim = r_sim
        self.q_sim = q_sim
        self.sig_sim = sig_sim
        self.r_opt = r_opt
        self.q_opt = q_opt
        self.sig_opt = sig_opt
        self.opt_type = opt_type
        self.model = model
        self.s_sim = SingleVarSimulation(n_scn, s_0, self.taus, r_sim, q_sim, sig_sim, model)

    def set_fix_commission(self, fixed_commission: float):
        self.FIX_COMMISSION = fixed_commission

    def set_var_commission(self, variable_commission: float):
        self.VAR_COMMISSION = variable_commission

    def set_reb_level(self, rebalance_level: float):
        self.REB_LEVEL = rebalance_level

    def tran_cost(self, n_shares):
        return self.FIX_COMMISSION + n_shares * self.VAR_COMMISSION

    def sim_to_term(self):
        # initial step
        s_curr = self.s_sim.s_curr
        v_curr = BSMVecS.price(s_curr, self.k, self.taus[0], self.r_opt, self.q_opt, self.sig_opt, self.opt_type)
        d_curr = BSMVecS.delta(s_curr, self.k, self.taus[0], self.r_opt, self.q_opt, self.sig_opt, self.opt_type)
        d_prev = d_curr.copy()
        cash_acc = v_curr - d_curr * s_curr
        reb_count = np.ones(self.s_sim.n_scn)

        # from second till last
        for idx, tau in enumerate(self.taus[1:]):
            dt = self.taus[idx] - tau
            inc_bond = exp(self.r_sim * dt)
            cash_acc *= inc_bond
            self.s_sim.marching(dt)
            s_curr = self.s_sim.s_curr.copy()
            s_prev = self.s_sim.s_prev.copy()
            # rebalancing indicator
            reb = np.abs(s_curr / s_prev - np.ones(self.s_sim.n_scn)) > self.REB_LEVEL
            d_curr[reb] = BSMVecS.delta(s_curr, self.k, tau, self.r_opt, self.q_opt, self.sig_opt, self.opt_type)[reb]
            cash_acc += (d_prev - d_curr) * s_curr
            d_prev = d_curr.copy()
            reb_count[reb] += np.ones(self.s_sim.n_scn)[reb]

        # only display the last one
        payoff = BSMVecS.payoff(s_curr, self.k, self.opt_type)
        pnl = payoff - cash_acc

        return pnl, reb_count
