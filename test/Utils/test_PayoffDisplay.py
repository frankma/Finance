import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from unittest import TestCase

from src.Utils.PayoffDisplay import PayoffDisplay
from src.Utils.Types.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestPayoffDisplay(TestCase):
    def test_payoff(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        s = np.linspace(10.0, 30.0)
        strike = 15.0
        opt_type = OptionType.call
        position = 1.0
        payoff = PayoffDisplay.payoff(strike, opt_type, position, s)
        plt.plot(s, payoff)
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_display(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
