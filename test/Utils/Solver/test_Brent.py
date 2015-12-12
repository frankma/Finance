from unittest import TestCase

import numpy as np

from src.Utils.Solver.Brent import Brent
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'


class TestBrent(TestCase):
    def test_solve_loc(self):
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
        print(res, res2)

        pass
