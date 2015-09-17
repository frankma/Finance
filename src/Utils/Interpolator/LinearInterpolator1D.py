import numpy as np

from src.Utils.Interpolator.IInterpolator1D import IInterpolator1D

__author__ = 'frank.ma'


class LinearInterpolator1D(IInterpolator1D):

    def __init__(self, x: list, y: list, extrapolate='flat'):
        super().__init__(x, y)
        self.ex = extrapolate

    def sanity_check(self):
        super().sanity_check()
        if self.ex not in ['flat', 'linear']:
            raise AttributeError('Unrecognised extrapolation type %s' % self.ex)

    def calc_loc(self, v: float):

        if self.x.__len__() == 0:
            raise Exception('Empty interpolation space, no value to return.')
        elif self.x.__len__() == 1:
            return self.y[0]
        else:
            if v < self.x[0]:
                if self.ex == 'flat':
                    return self.y[0]
                elif self.ex == 'linear':
                    return self.y[0] + (v - self.x[1]) / (self.x[0] - self.x[1]) * (self.y[0] - self.y[1])
            elif v > self.x[-1]:
                if self.ex == 'flat':
                    return self.y[-1]
                elif self.ex == 'linear':
                    return self.y[-1] + (v - self.x[-2]) / (self.x[-1] - self.x[-2]) * (self.y[-1] - self.y[-2])
            else:
                i = 0
                while v < self.x[i]:
                    i += 1
                return self.y[i] + (v - self.x[i]) / (self.x[i + 1] - self.x[i]) * (self.y[i + 1] - self.y[i])

    def calc(self, v: float):
        if (v < self.x[0] or v > self.x[0]) and self.ex == 'lin':
            print("WARNING: requested linear extrapolation while only flat extrapolation is applied.")
        return np.interp([v], self.x, self.y)
