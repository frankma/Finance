from scipy.optimize import brentq

from src.Utils.Solver.ISolver import ISolver
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'


class Brent(ISolver):

    def __init__(self, f: IUnivariateFunction, lower_bound: float, upper_bound: float):
        self.f = f
        self.lb = lower_bound
        self.ub = upper_bound

    def solve_loc(self):
        # TODO: develop a local version for comparison
        pass

    def solve(self):
        x = brentq(self.f.evaluate, self.lb, self.ub, xtol=self.ABS_TOL, rtol=self.REL_TOL, maxiter=self.ITR_TOL)
        return x
