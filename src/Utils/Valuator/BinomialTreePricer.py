import numpy as np

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BinomialTreePricer(object):
    def __init__(self, s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType,
                 n_steps: int = 99):
        self.s = s
        self.k = k
        self.tau = tau
        self.r = r
        self.q = q
        self.sig = sig
        self.opt_type = opt_type
        self.n_steps = n_steps
        # common featured calculation
        self.dt = self.tau / self.n_steps
        self.db = np.exp(-self.r * self.dt)
        self.u, self.d, self.p_u = self.calc_u_d_p(self.dt, self.r, self.q, self.sig)
        self.p_d = 1.0 - self.p_u
        pass

    @staticmethod
    def calc_u_d_p(dt: float, r: float, q: float, sig: float):
        up = np.exp(sig * np.sqrt(dt))
        down = 1.0 / up
        prob_up = (np.exp((r - q) * dt) - down) / (up - down)
        return up, down, prob_up

    @staticmethod
    def create_final_state(s: float, n_steps: int, up: float):
        n_ups = np.linspace(n_steps, 0, num=n_steps + 1)
        n_downs = np.linspace(0, n_steps, num=n_steps + 1)
        increments = np.subtract(n_ups, n_downs)
        return s * np.array(np.power(up, increments))

    @staticmethod
    def calc_intrinsic_value(s_vec: np.array, k: float, opt_type: OptionType):
        eta = opt_type.value
        zeros = np.zeros(np.shape(s_vec))
        return np.maximum(eta * (s_vec - k), zeros)

    def price_eur_opt(self):
        state = self.create_final_state(self.s, self.n_steps, self.u)
        option = self.calc_intrinsic_value(state, self.k, self.opt_type)
        for idx in range(self.n_steps, 0, -1):
            option = self.db * (self.p_u * option[:-1] + self.p_d * option[1:])
        if option.__len__() != 1:
            raise ValueError('unexpected shape of final result')
        return option[0]

    def price_ame_opt(self):
        state = self.create_final_state(self.s, self.n_steps, self.u)
        option = self.calc_intrinsic_value(state, self.k, self.opt_type)
        for idx in range(self.n_steps, 0, -1):
            state = self.p_u * state[:-1] + self.p_d * state[1:]
            option = self.db * (self.p_u * option[:-1] + self.p_d * option[1:])
            intrinsic = self.calc_intrinsic_value(state, self.k, self.opt_type)
            option = np.maximum(intrinsic, option)
        if option.__len__() != 1:
            raise ValueError('unexpected shape of final result')
        return option[0]
