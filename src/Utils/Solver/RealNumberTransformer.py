from math import tan, atan, pi, exp, log
from scipy.stats import norm


__author__ = 'frank.ma'


class RealNumberTransformer(object):

    def __init__(self, l: float or None, u: float or None, method='abs'):
        self.l = l
        self.u = u
        if l is not None and u is not None:
            self.rgn = u - l
        self.method = method

    def uc_to_c(self, unconstraint: float):

        constraint = unconstraint

        if self.l is None and self.u is None:
            print('WARNING: neither lower nor upper bound is provided, no transform is performed.')
        elif self.l is None:
            constraint = self.u - exp(unconstraint)
        elif self.u is None:
            constraint = self.l + exp(unconstraint)
        else:
            if self.method == 'abs':
                constraint = self.l + 0.5 * self.rgn * (1.0 + unconstraint / (1 + abs(unconstraint)))
            elif self.method == 'tan':
                constraint = self.l + self.rgn * (atan(unconstraint) / pi + 0.5)
            elif self.method == 'norm':
                constraint = self.l + self.rgn * norm.cdf(unconstraint)
            else:
                raise Exception('unrecognized transformation method')

        return constraint

    def c_to_uc(self, constraint: float):

        unconstraint = constraint

        if self.l is None and self.u is None:
            print('WARNING: neither lower nor upper bound is provided, no transform is performed.')
        elif self.l is None:
            unconstraint = log(self.u - constraint)
        elif self.u is None:
            unconstraint = log(constraint - self.l)
        else:
            if self.method == 'abs':
                mid = 0.5 * (self.l + self.u)
                if constraint < mid:
                    unconstraint = 1.0 - 0.5 * self.rgn / (constraint - self.l)
                else:
                    unconstraint = -1.0 - 0.5 * self.rgn / (constraint - self.u)
            elif self.method == 'tan':
                unconstraint = tan(((constraint - self.l) / self.rgn - 0.5) * pi)
            elif self.method == 'norm':
                unconstraint = norm.ppf(constraint - self.l) / self.rgn
            else:
                raise Exception("unrecognized transformation method")

        return unconstraint
