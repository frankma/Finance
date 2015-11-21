import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval

__author__ = 'frank.ma'


class PolynomialExtrapolator(object):
    """
    Extrapolator adopts polynomial function with formula of
    x^{lam} * e^{\sum_0^n{beta_i * x^{eta * i}}}, where eta ~ -1 or 1
    """

    def __init__(self, lam: float, betas: np.array, eta: int):
        self.lam = lam
        self.order = betas.__len__()
        self.betas = betas
        self.eta = eta

    def extrapolate(self, xs: np.array):
        return self.zero_order(xs, self.betas, self.lam, self.eta)

    @staticmethod
    def __get_base(xs: np.array, eta: int):
        if eta == 1:
            base = xs
        elif eta == -1:
            base = np.reciprocal(xs)
        else:
            raise ValueError('unrecognized eta (%i), it must be either -1 or 1' % eta)
        return base

    @staticmethod
    def __get_poly_derivative(xs: np.array, betas: np.array, order: int = 1):
        poly = np.poly1d(betas)
        derivative = np.polyder(poly, m=order)
        return derivative(xs)

    @staticmethod
    def zero_order(xs: np.array, betas: np.array, lam: float, eta: int):
        base = PolynomialExtrapolator.__get_base(xs, eta)
        zero = np.power(xs, eta * lam) * np.exp(polyval(base, betas))
        return zero

    @staticmethod
    def first_order(xs: np.array, betas: np.array, lam: float, eta: int):
        base = PolynomialExtrapolator.__get_base(xs, eta)
        zero = PolynomialExtrapolator.zero_order(base, betas, lam, eta)
        factor = lam / base + PolynomialExtrapolator.__get_poly_derivative(base, betas, order=1)
        if eta == -1:
            factor *= -xs ** 2
        first = zero * factor
        return first

    @staticmethod
    def second_order(xs: np.array, betas: np.array, lam: float, eta: int):
        base = xs
        zero = PolynomialExtrapolator.zero_order(base, betas, lam, eta)
        first = PolynomialExtrapolator.first_order(base, betas, lam, eta)
        factor = -lam / (base ** 2) + PolynomialExtrapolator.__get_poly_derivative(base, betas, order=2)
        return (first ** 2) / zero + zero * factor
