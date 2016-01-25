import logging
import sys
from unittest import TestCase

import numpy as np

from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestBrent(TestCase):
    def test_solve_loc(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        class _PolyMin(IUnivariateFunction):
            def __init__(self, coefficients: np.array or list):
                self.coefficients = coefficients
                self.poly = np.polynomial.Polynomial(self.coefficients)

            def evaluate(self, x):
                return self.poly(x)

        third_order_poly = _PolyMin([3.0, -5.0, 1.0, 1.0])
        bt = Brent(third_order_poly, -4.0, 4.0 / 3.0)
        res = bt.solve()
        res2 = bt.solve_loc()
        print(res, res2, res - res2)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
