import numpy as np

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BinomialTreePricer(object):
    @staticmethod
    def calc_u_and_d(dt: float, sig: float):
        u = np.exp(sig * np.sqrt(dt))
        return u, 1.0 / u

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

    @staticmethod
    def price_european_option(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType,
                              n_steps: int = 99):
        dt = tau / float(n_steps)
        u, d = BinomialTreePricer.calc_u_and_d(dt, sig)
        disc = np.exp(-r * dt)
        drift = np.exp((r - q) * dt)
        p_u = (drift - d) / (u - d)
        p_d = 1.0 - p_u

        state = BinomialTreePricer.create_final_state(s, n_steps, u)
        option = BinomialTreePricer.calc_intrinsic_value(state, k, opt_type)
        for idx in range(n_steps, 0, -1):
            option = disc * (p_u * option[:-1] + p_d * option[1:])
        if option.__len__() != 1:
            raise ValueError('unexpected shape of final result')
        return option[0]

    @staticmethod
    def price_american_option(s: float, k: float, tau: float, r: float, q: float, sig: float, opt_type: OptionType,
                              n_steps: int = 99):
        dt = tau / float(n_steps)
        u, d = BinomialTreePricer.calc_u_and_d(dt, sig)
        disc = np.exp(-r * dt)
        drift = np.exp((r - q) * dt)
        p_u = (drift - d) / (u - d)
        p_d = 1.0 - p_u

        state = BinomialTreePricer.create_final_state(s, n_steps, u)
        option = BinomialTreePricer.calc_intrinsic_value(state, k, opt_type)
        for idx in range(n_steps, 0, -1):
            state = (1.0 / drift) * (p_u * state[:-1] + p_d * state[1:])
            option = disc * (p_u * option[:-1] + p_d * option[1:])
            intrinsic = BinomialTreePricer.calc_intrinsic_value(state, k, opt_type)
            option = np.maximum(intrinsic, option)
        if option.__len__() != 1:
            raise ValueError('unexpected shape of final result')
        return option[0]
