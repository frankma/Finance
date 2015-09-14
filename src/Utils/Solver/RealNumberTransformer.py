from math import tan, atan, pi
from scipy.stats import norm


__author__ = 'frank.ma'


class RealNumberTransformer(object):

    def __init__(self, l: float, u: float, method='abs'):
        self.l = l
        self.u = u
        self.rgn = u - l
        self.method = method

    def uc_to_c(self, unconstraint: float):

        if self.method == 'abs':
            return self.l + 0.5 * self.rgn * (1.0 + unconstraint / (1 + abs(unconstraint)))
        elif self.method == 'tan':
            return self.l + self.rgn * (atan(unconstraint) / pi + 0.5)
        elif self.method == 'norm':
            return self.l + self.rgn * norm.cdf(unconstraint)
        else:
            raise Exception("unrecognized transformation method")

    def c_to_uc(self, constraint: float):

        if self.method == 'abs':
            mid = 0.5 * (self.l + self.u)
            if constraint < mid:
                return 1.0 - 0.5 * self.rgn / (constraint - self.l)
            else:
                return -1.0 - 0.5 * self.rgn / (constraint - self.u)
        elif self.method == 'tan':
            return tan(((constraint - self.l) / self.rgn - 0.5) * pi)
        elif self.method == 'norm':
            return norm.ppf(constraint - self.l) / self.rgn
        else:
            raise Exception("unrecognized transformation method")
