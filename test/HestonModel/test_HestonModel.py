import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt

from src.HestonModel.HestonModel import HestonModel

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestHestonModel(TestCase):
    def test_sim_forward_den(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        mu = 0.01
        v_0 = 0.4
        kappa = 0.02
        theta = 0.02
        nu = 0.3
        rho = 0.25

        model = HestonModel(mu, v_0, kappa, theta, nu, rho)
        spot = 150.0
        t = 0.25
        den, bins = model.sim_forward_den(spot, t)

        plt.plot(bins, den)
        plt.show()

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
