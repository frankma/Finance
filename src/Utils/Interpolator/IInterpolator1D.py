__author__ = 'frank.ma'


class IInterpolator1D(object):

    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y
        self.sanity_check()

    def sanity_check(self):
        if self.x.__len__() != self.y.__len__():
            raise ValueError('Interpolator inputs are not in the same length.')
        if self.x.__len__() > 1 and not all(a < b for a, b in zip(self.x[:-1], self.x[1:])):
            raise ValueError('Interpolator x input need to be monotonicity')

    def calc(self, v: float):
        pass
