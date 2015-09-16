from math import tan, atan, pi, exp, log
from scipy.stats import norm


__author__ = 'frank.ma'


class RealNumberTransformer(object):

    def __init__(self, lower_bound: float or None, upper_bound: float or None, method='abs'):
        self.lb = lower_bound
        self.ub = upper_bound
        if lower_bound is not None and upper_bound is not None:
            self.distance = upper_bound - lower_bound
        self.method = method

    def uc_to_c(self, unconstraint: float):

        constraint = unconstraint

        if self.lb is None and self.ub is None:
            print('WARNING: neither lower nor upper bound is provided, no transform will be performed.')
        elif self.lb is None:
            constraint = self.ub - exp(unconstraint)
        elif self.ub is None:
            constraint = self.lb + exp(unconstraint)
        else:
            if self.method == 'abs':
                constraint = self.lb + 0.5 * self.distance * (1.0 + unconstraint / (1 + abs(unconstraint)))
            elif self.method == 'tan':
                constraint = self.lb + self.distance * (atan(unconstraint) / pi + 0.5)
            elif self.method == 'norm':
                constraint = self.lb + self.distance * norm.cdf(unconstraint)
            else:
                raise Exception('Unrecognized transformation method %s' % self.method)

        return constraint

    def c_to_uc(self, constraint: float):

        unconstraint = constraint

        if self.lb is None and self.ub is None:
            print('WARNING: neither lower nor upper bound is provided, no transform will be performed.')
        elif self.lb is None:
            unconstraint = log(self.ub - constraint)
        elif self.ub is None:
            unconstraint = log(constraint - self.lb)
        else:
            if self.method == 'abs':
                mid = 0.5 * (self.lb + self.ub)
                if constraint < mid:
                    unconstraint = 1.0 - 0.5 * self.distance / (constraint - self.lb)
                else:
                    unconstraint = -1.0 - 0.5 * self.distance / (constraint - self.ub)
            elif self.method == 'tan':
                unconstraint = tan(((constraint - self.lb) / self.distance - 0.5) * pi)
            elif self.method == 'norm':
                unconstraint = norm.ppf(constraint - self.lb) / self.distance
            else:
                raise Exception("Unrecognized transformation method %s" % self.method)

        return unconstraint
