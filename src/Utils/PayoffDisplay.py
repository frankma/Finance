import logging
import numpy as np
import matplotlib.pyplot as plt

from src.Utils.Types.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class PayoffDisplay(object):
    def __init__(self, strikes: list, opt_types: list, positions: list):
        self.strikes = strikes
        self.opt_types = opt_types
        self.positions = positions
        pass

    @staticmethod
    def payoff(strike: float, opt_type: OptionType, position: float, s: np.array):
        eta = opt_type.value
        zeros = np.zeros(np.shape(s))
        return position * np.maximum(eta * np.array(s - strike), zeros)

    def display(self):
        k_min = min(self.strikes)
        k_max = max(self.strikes)
        rgn = k_max - k_min
        s = np.array(np.linspace(k_min - 0.5 * rgn, k_max + 0.5 * rgn, num=200))
        payoffs = np.zeros(np.shape(s))
        for strike, opt_type, position in zip(self.strikes, self.opt_types, self.positions):
            payoff = self.payoff(strike, opt_type, position, s)
            payoffs += payoff
            plt.plot(s, payoff, '--', label=('%.1f %s %.2f' % (position, opt_type.name, strike)))
        plt.plot(s, payoffs, label='total')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        pass

