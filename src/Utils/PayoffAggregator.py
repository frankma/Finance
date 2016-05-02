import logging

import matplotlib.pyplot as plt
import numpy as np

from src.Utils.Types.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class PayoffAggregator(object):
    def __init__(self, strikes: list, opt_types: list, positions: list):
        self.strikes = strikes
        self.opt_types = opt_types
        self.positions = positions
        self.sop = zip(strikes, opt_types, positions)
        self.n = self.strikes.__len__()
        pass

    @staticmethod
    def payoff(strike: float, opt_type: OptionType, position: float, s: np.array):
        eta = opt_type.value
        zeros = np.zeros(np.shape(s))
        return position * np.maximum(eta * np.array(s - strike), zeros)

    def aggregate(self, size: int = 200):
        k_min = min(self.strikes)
        k_max = max(self.strikes)
        rgn = k_max - k_min
        s = np.array(np.linspace(k_min - 0.5 * rgn, k_max + 0.5 * rgn, num=size))
        payoffs = np.zeros((self.n, size))
        labels = [' ' for _ in range(self.n)]
        for idx, (strike, opt_type, position) in enumerate(self.sop):
            payoff = self.payoff(strike, opt_type, position, s)
            payoffs[idx, :] = payoff
            labels[idx] = '%.1f %s %.2f' % (position, opt_type.name, strike)
        return s, np.sum(payoffs, axis=0), payoffs, labels

    def display(self):
        s, agg, pff, lbs = self.aggregate()
        for idx in range(self.n):
            payoff = pff[idx, :]
            plt.plot(s, payoff, '--', label=lbs[idx])
        plt.plot(s, agg, label='total')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
