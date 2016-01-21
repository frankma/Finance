import logging

from scipy.optimize import brentq

from src.Utils.Solver.ISolver import ISolver
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Brent(ISolver):
    def __init__(self, f: IUnivariateFunction, a: float, b: float):
        v_lb = f.evaluate(a)
        v_ub = f.evaluate(b)
        if v_lb * v_ub > 0.0:
            raise ValueError('Initial initial bucket: function evaluation yield same sign.')
        self.f = f
        self.a = a
        self.b = b

    def solve_loc(self):
        a, b = self.a, self.b
        v_a, v_b = self.f.evaluate(a), self.f.evaluate(b)

        if abs(v_a) < abs(v_b):
            # swap position for a faster convergence speed
            a, b = b, a
            v_a, v_b = v_b, v_a
        c = self.a
        d = c
        v_c = v_a
        flg = True
        iterator_count = 0

        while (abs(v_b) > self.ABS_TOL) or abs(b - a) > self.ABS_TOL:
            iterator_count += 1
            if abs(v_a - v_c) > self.ABS_TOL and abs(v_b - v_c) > self.ABS_TOL:
                s = self.inverse_quadratic_method(self.f.evaluate, a, b, c)
            else:
                s = self.secant_method(self.f.evaluate, a, b)

            condition_1 = not (3.0 * a + b) / 4.0 < s < b
            condition_2 = flg and abs(s - b) >= abs(b - c) / 2.0
            condition_3 = (not flg) and abs(s - b) >= abs(c - d) / 2.0
            condition_4 = flg and abs(b - c) < self.ABS_TOL
            condition_5 = (not flg) and abs(c - d) < self.ABS_TOL

            if condition_1 or condition_2 or condition_3 or condition_4 or condition_5:
                s = self.bisection_method(a, b)
                flg = True
            else:
                flg = False

            d = c
            c = b

            v_s = self.f.evaluate(s)
            v_a = self.f.evaluate(a)
            v_b = self.f.evaluate(b)
            v_c = self.f.evaluate(b)

            if v_a * v_s < 0.0:
                b, v_b = s, v_s
            else:
                a, v_a = s, v_s

            if abs(v_a) < abs(v_b):
                a, b = b, a

            if iterator_count > self.ITR_TOL:
                logger.warning('maximum iteration count (%i) reached' % self.ITR_TOL)
                break

        return b

    def solve(self):
        x = brentq(self.f.evaluate, self.a, self.b, xtol=self.ABS_TOL, rtol=self.REL_TOL, maxiter=self.ITR_TOL)
        return x

    @staticmethod
    def bisection_method(a: float, b: float):
        return 0.5 * (a + b)

    @staticmethod
    def secant_method(func, a: float, b: float):
        v_a = func(a)
        v_b = func(b)
        return b - v_b * (b - a) / (v_b - v_a)

    @staticmethod
    def inverse_quadratic_method(func, a: float, b: float, c: float):
        v_a = func(a)
        v_b = func(b)
        v_c = func(c)
        term_1 = a * v_b * v_c / (v_a - v_b) / (v_a - v_c)
        term_2 = b * v_a * v_c / (v_b - v_a) / (v_b - v_c)
        term_3 = c * v_a * v_b / (v_c - v_a) / (v_c - v_b)
        return term_1 + term_2 + term_3
