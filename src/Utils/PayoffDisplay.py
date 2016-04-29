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
        s = np.linspace(k_min - 0.5 * rgn, k_max + 0.5 * rgn, num=200)
        pass
