from math import sqrt, exp

import numpy as np
from scipy.stats import norm

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BSMSpotVec(object):

    @staticmethod
    def price(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):

        eta = opt_type.value

        if tau < 1e-12:
            if tau < 0.0:
                print('WARNING: Negative time to expiry is given, payoff will be returned.')
            return BSMSpotVec.payoff(s, k, opt_type)
        else:
            d_1 = (np.log(s / k) + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt(tau))
            d_2 = d_1 - sig * sqrt(tau)
            disc_r = exp(-r * tau)
            disc_q = exp(-q * tau)
            return eta * (disc_q * s * norm.cdf(eta * d_1) - disc_r * k * norm.cdf(eta * d_2))

    @staticmethod
    def delta(s: np.array, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType):

        eta = opt_type.value

        if tau < 1e-12:
            if tau < 0.0:
                print('WARNING: Negative time to expiry is given, zero delta will be returned.')
            return np.zeros(s.__len__())
        else:
            d_1 = (np.log(s / k) + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt(tau))
            disc_q = exp(-q * tau)
            return eta * disc_q * norm.cdf(eta * d_1)

    @staticmethod
    def payoff(s: np.array, k: float, opt_type: OptionType):
        return np.maximum(opt_type.value * (s - k), np.zeros(s.__len__()))


class Simulation(object):

    @staticmethod
    def evolve(s_prev: np.array, dt: float, r: float, q: float, sig: float, model='lognormal'):

        rand_norm = norm.ppf(np.random.random(s_prev.__len__()))

        if model == 'lognormal':
            return s_prev * np.exp((r - q - 0.5 * sig**2) * dt + sig * rand_norm * sqrt(dt))
        elif model == 'normal':
            return s_prev * (1.0 + (r - q) * dt + sig * rand_norm * sqrt(dt))
        else:
            raise ValueError('Unrecognized model input: %s' % model)

    def __init__(self, n_scn: int, s_0: float, taus: list, r: float, q: float, sig: float, model='lognormal'):
        self.n_scn = n_scn
        self.taus = taus
        self.s_0 = s_0
        self.r = r
        self.q = q
        self.sig = sig
        self.model = model

        self.s_curr = np.ones(self.n_scn) * self.s_0
        self.s_prev = np.zeros(self.n_scn)

    def marching(self, dt: float):
        self.s_prev = self.s_curr.copy()
        self.s_curr = self.evolve(self.s_prev, dt, self.r, self.q, self.sig, self.model)
        return self.s_curr


class DeltaHedging(object):

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

    def sim_to_terminal(self):

        s_sim = Simulation(self.n_scn, self.s_0, self.taus, self.r_sim, self.q_sim, self.sig_sim, self.model)

        # initial step
        s_curr = s_sim.s_curr
        v_curr = BSMSpotVec.price(s_curr, self.k, self.taus[0], self.r_opt, self.q_opt, self.sig_opt, self.opt_type)
        d_curr = BSMSpotVec.delta(s_curr, self.k, self.taus[0], self.r_opt, self.q_opt, self.sig_opt, self.opt_type)
        d_prev = d_curr.copy()
        cash_acc = v_curr - d_curr * s_curr
        print(cash_acc)

        for idx, tau in enumerate(self.taus[1:]):
            dt = self.taus[idx] - tau
            inc_bond = exp(self.r_sim * dt)
            cash_acc *= inc_bond
            s_curr = s_sim.marching(dt)
            d_curr = BSMSpotVec.delta(s_curr, self.k, self.taus[0], self.r_opt, self.q_opt, self.sig_opt, self.opt_type)
            cash_acc += (d_prev - d_curr) * s_curr
            d_prev = d_curr.copy()

        payoff = BSMSpotVec.payoff(s_curr, self.k, self.opt_type)
        pnl = cash_acc - payoff

        return pnl, cash_acc, payoff


if __name__ == '__main__':

    N_SCN = 2
    N_STP = 10
    S_0 = 100
    K = 100
    TAU = 1.0
    R_SIM = 0.0
    Q_SIM = 0.0
    SIG_SIM = 0.0000000002
    R_OPT = R_SIM
    Q_OPT = Q_SIM
    SIG_OPT = SIG_SIM
    OPT_TYPE = OptionType.call

    dh = DeltaHedging(N_SCN, N_STP, S_0, K, TAU, R_SIM, Q_SIM, SIG_SIM, R_OPT, Q_OPT, SIG_OPT, OPT_TYPE, 'lognormal')
    p_n_l, c_a, pff = dh.sim_to_terminal()
    print(p_n_l, c_a, pff)
    print(np.average(p_n_l), np.std(p_n_l))
