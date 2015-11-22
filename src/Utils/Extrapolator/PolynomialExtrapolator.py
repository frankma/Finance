import numpy as np
from numpy.polynomial.polynomial import polyval

__author__ = 'frank.ma'


class PolynomialExtrapolator(object):
    """
    Extrapolator adopts polynomial function with formula of
    x^{lam} * e^{\sum_0^n{beta_i * x^{eta * i}}}, where eta ~ -1 or 1
    """

    def __init__(self, lam: float, betas: np.array, is_reciprocal: bool):
        self.lam = lam
        self.order = betas.__len__()
        self.betas = betas
        self.is_reciprocal = is_reciprocal

    def extrapolate(self, xs: np.array):
        return self.zeroth_order(xs, self.betas, self.lam, self.is_reciprocal)

    @staticmethod
    def __get_base(xs: np.array, is_reciprocal: bool):
        if is_reciprocal:
            return np.reciprocal(xs)
        else:
            return xs

    @staticmethod
    def __get_poly_derivative(xs: np.array, betas: np.array, order: int = 1):
        poly = np.poly1d(betas[::-1])
        derivative = np.polyder(poly, m=order)
        return derivative(xs)

    @staticmethod
    def zeroth_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool):
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = np.power(base, lam) * np.exp(polyval(base, betas))
        return zeroth

    @staticmethod
    def first_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool):
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = PolynomialExtrapolator.zeroth_order(base, betas, lam, False)
        first = zeroth * (lam / base + PolynomialExtrapolator.__get_poly_derivative(base, betas, order=1))
        if is_reciprocal:
            first *= -base ** 2
        return first

    @staticmethod
    def second_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool):
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = PolynomialExtrapolator.zeroth_order(base, betas, lam, False)
        first = PolynomialExtrapolator.first_order(base, betas, lam, False)
        second = (first ** 2) / zeroth
        second += zeroth * (PolynomialExtrapolator.__get_poly_derivative(base, betas, order=2) - lam / (base ** 2))
        if is_reciprocal:
            second *= base ** 4
            second += first * 2.0 * (base ** 3)
        return second
