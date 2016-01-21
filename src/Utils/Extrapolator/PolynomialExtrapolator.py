import logging

import numpy as np
from numpy.polynomial.polynomial import polyval

from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


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

    # fit to the wing quote on the volatility surface, assume lower strikes are put quotes and higher strikes are calls
    @staticmethod
    def quadratic_fit(lam: float, strike: float, price: float, delta_k: float, gamma_k: float, opt_type: OptionType):
        # since always consider OTM options as the prices move towards zero, put for low strike and call for the other.
        is_reciprocal = opt_type == OptionType.call
        base = PolynomialExtrapolator.__get_base(strike, is_reciprocal)
        first = delta_k
        second = gamma_k
        # term need to be cancelled if reciprocal as derivative is on the base but not the strike
        if is_reciprocal:
            first /= -base ** 2
            second -= first * 2.0 * (base ** 3)
            second /= base ** 4
        beta_2 = 0.5 * (second / price - ((first / price) ** 2) + lam / (base ** 2))
        beta_1 = first / price - lam / base - 2.0 * beta_2 * base
        # beta_1 = (first - second * base) / price - 2.0 * lam / base + (first / price) ** 2 * base
        beta_0 = np.log(price / (base ** lam)) - beta_1 * base - beta_2 * (base ** 2)
        # beta_0 = np.log(price) - lam * np.log(base) - first / price * base \
        #          + 0.5 * second * (base ** 2) / price + 1.5 * lam - 0.5 * ((first * base / price) ** 2)
        betas = np.array([beta_0, beta_1, beta_2])
        return PolynomialExtrapolator(lam, betas, is_reciprocal)

    @staticmethod
    def __get_base(xs: np.array, is_reciprocal: bool):
        if is_reciprocal:
            return np.reciprocal(xs)
        else:
            return xs

    @staticmethod
    def __get_poly_derivative(xs: np.array, betas: np.array, order: int = 1):
        poly = np.poly1d(betas[::-1])  # poly1d takes polynomial starts from high order to low
        derivative = np.polyder(poly, m=order)
        return derivative(xs)

    @staticmethod
    def zeroth_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool) -> np.array:
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = np.power(base, lam) * np.exp(polyval(base, betas))
        return zeroth

    @staticmethod
    def first_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool) -> np.array:
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = PolynomialExtrapolator.zeroth_order(base, betas, lam, False)
        first = zeroth * (lam / base + PolynomialExtrapolator.__get_poly_derivative(base, betas, order=1))
        # first order chain rule: ∂ V / ∂ X = ∂ V / ∂ (1 / X)  / (-X^2)
        if is_reciprocal:
            first *= -base ** 2
        return first

    @staticmethod
    def second_order(xs: np.array, betas: np.array, lam: float, is_reciprocal: bool) -> np.array:
        base = PolynomialExtrapolator.__get_base(xs, is_reciprocal)
        zeroth = PolynomialExtrapolator.zeroth_order(base, betas, lam, False)
        first = PolynomialExtrapolator.first_order(base, betas, lam, False)
        second = (first ** 2) / zeroth
        second += zeroth * (PolynomialExtrapolator.__get_poly_derivative(base, betas, order=2) - lam / (base ** 2))
        # second order chain rule: ∂^2 V / ∂ X^2 = ∂^2 V / ∂ (1 / X)^2 / X^4 + ∂ V / ∂ (1 / X) * 2 / X^3
        if is_reciprocal:
            second *= base ** 4
            second += first * 2.0 * (base ** 3)
        return second
