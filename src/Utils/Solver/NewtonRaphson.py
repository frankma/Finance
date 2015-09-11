from src.Utils.Solver.ISolver import ISolver
from src.Utils.Solver.IVariateFunction import IUnivariateFunction

__author__ = 'frank.ma'


class NewtonRaphson(ISolver):

    def __init__(self, f: IUnivariateFunction, d: IUnivariateFunction, init_guess: float):
        self.f = f
        self.d = d
        self.init_guess = init_guess

    def solve(self):
        x = self.init_guess
        y = self.f.evaluate(x)
        d = self.d.evaluate(x)
        count = 0

        while count < self.ITR_TOL and abs(y) > self.ABS_TOL:
            count += 1
            x -= y / d
            y = self.f.evaluate(x)
            d = self.d.evaluate(x)

        if count >= self.ITR_TOL:
            print("WARNING: Newton-Raphson method iteration maxed out %i steps, return best guess with error %r"
                  % (self.ITR_TOL, y))
        return x