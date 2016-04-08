import logging
import sys

import numpy as np

from unittest import TestCase

from src.Utils.Solver.SimpleLinearRegression import SimpleLinearRegression

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestSimpleLinearRegression(TestCase):
    def test_regress(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        x = np.linspace(-10.0, 10.0)
        y = np.linspace(12.0, -12.0)
        alpha, beta = SimpleLinearRegression.regress(x, y)
        self.assertAlmostEqual(alpha, 0.0)
        self.assertAlmostEqual(beta, -10 / 12.0)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_regress_np(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        rnd_x = np.random.randint(-10, 10, size=100)
        rnd_y = np.random.randint(-20, 20, size=100)
        x = np.cumsum(rnd_x)
        y = np.cumsum(rnd_y)
        alpha_sf, beta_sf = SimpleLinearRegression.regress(x, y)
        alpha_np, beta_np = SimpleLinearRegression.regress_np(x, y)
        # TODO: rewrite unite test
        print(alpha_sf, alpha_np)
        print(beta_sf, beta_np)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
