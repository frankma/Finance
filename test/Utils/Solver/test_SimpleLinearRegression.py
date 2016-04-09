import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

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
        self.assertAlmostEqual(beta, -12 / 10.0)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_regress_visual(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        alpha = 0.5
        beta = 2.5
        err = 0.2
        x = np.array(np.linspace(1, 9))
        y = alpha + beta * x + np.random.normal(size=x.__len__()) * err

        slr = SimpleLinearRegression(x, y)

        plt.scatter(x, y)
        plt.plot([x.min(), x.max()], [slr.predict(x.min()), slr.predict(x.max())])
        plt.show()

        # comment out for just visual check in plots
        # self.assertAlmostEqual(alpha, slr.alpha, places=1)
        # self.assertAlmostEqual(beta, slr.beta, places=1)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_regress_np(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        alpha = 0.01
        beta = 2.1
        err = 1.0
        x = np.array(np.linspace(1, 9))
        y = alpha + beta * x + np.random.normal(size=x.__len__()) * err

        interception_loc, slope_loc = SimpleLinearRegression.regress(x, y)
        interception_np, slope_np = SimpleLinearRegression.regress_np(x, y)

        self.assertAlmostEqual(interception_loc, interception_np, places=12, msg='interception reconciliation failed')
        self.assertAlmostEqual(slope_loc, slope_np, places=12, msg='slope reconciliation failed')

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
