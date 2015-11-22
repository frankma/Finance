from unittest import TestCase

import numpy as np

from src.Utils.Extrapolator.PolynomialExtrapolator import PolynomialExtrapolator

__author__ = 'frank.ma'


class TestPolynomialExtrapolator(TestCase):
    def test_first_order(self):
        xs = np.linspace(10.0, 200.0, num=191)
        dx = 1e-6
        lam = 0.5
        betas = np.array([1.0, -2.0, 3.0, -4.0]) * 1.0e-6

        for is_reciprocal in [False, True]:
            xs_d = xs * (1.0 - dx)
            xs_u = xs * (1.0 + dx)
            v_d = PolynomialExtrapolator.zeroth_order(xs_d, betas, lam, is_reciprocal)
            v_u = PolynomialExtrapolator.zeroth_order(xs_u, betas, lam, is_reciprocal)
            first_n = (v_u - v_d) / (xs_u - xs_d)
            first = PolynomialExtrapolator.first_order(xs, betas, lam, is_reciprocal)
            for idx in range(xs.__len__()):
                rel_diff = first[idx] / first_n[idx] - 1.0
                self.assertAlmostEqual(rel_diff, 0.0, places=6)
                # print('Is reciprocal: ', is_reciprocal, '\nAnalytical', first, '\nNumerical', first_n)
        pass

    def test_second_order(self):
        xs = np.linspace(10.0, 200.0, num=191)
        dx = 1e-4
        lam = 0.5
        betas = np.array([1.0, -2.0, 3.0, -4.0]) * 1e-6

        for is_reciprocal in [False, True]:
            xs_d = xs * (1.0 - dx)
            xs_u = xs * (1.0 + dx)
            v_d = PolynomialExtrapolator.zeroth_order(xs_d, betas, lam, is_reciprocal)
            v = PolynomialExtrapolator.zeroth_order(xs, betas, lam, is_reciprocal)
            v_u = PolynomialExtrapolator.zeroth_order(xs_u, betas, lam, is_reciprocal)
            second_n = (v_u - 2.0 * v + v_d) / ((0.5 * (xs_u - xs_d)) ** 2)
            second = PolynomialExtrapolator.second_order(xs, betas, lam, is_reciprocal)
            for idx in range(xs.__len__()):
                rel_diff = second[idx] / second_n[idx] - 1.0
                self.assertAlmostEqual(rel_diff, 0.0, places=4)
                # print('Is reciprocal: ', is_reciprocal, '\nAnalytical', second, '\nNumerical', second_n)
        pass

    def test_eta(self):
        xs = np.linspace(10.0, 50.0, num=2)
        xs_rec = np.reciprocal(xs)
        lam = 0.5
        betas = np.array([1.0, -2.0, 3.0, -4.0, 5.0]) / 1e6

        zeroth = PolynomialExtrapolator.zeroth_order(xs, betas, lam, False)
        zeroth_rec = PolynomialExtrapolator.zeroth_order(xs_rec, betas, lam, True)

        first = PolynomialExtrapolator.first_order(xs, betas, lam, False)
        first_rec = PolynomialExtrapolator.first_order(xs_rec, betas, lam, True)
        # w.r.t xs instead of xs reciprocal, apply first order chain rule
        first_rec *= -xs_rec ** 2

        second = PolynomialExtrapolator.second_order(xs, betas, lam, False)
        second_rec = PolynomialExtrapolator.second_order(xs_rec, betas, lam, True)
        # w.r.t xs instead of xs reciprocal, apply second order chain rule
        second_rec *= xs_rec ** 4
        second_rec -= 2.0 * first_rec * xs_rec
        for idx in range(xs.__len__()):
            self.assertAlmostEqual(zeroth[idx], zeroth_rec[idx], places=12)
            self.assertAlmostEqual(first[idx], first_rec[idx], places=12)
            self.assertAlmostEqual(second[idx], second_rec[idx], places=12)
        pass
