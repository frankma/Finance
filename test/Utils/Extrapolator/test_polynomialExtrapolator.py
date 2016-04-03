import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.Extrapolator.PolynomialExtrapolator import PolynomialExtrapolator
from src.Utils.Types.OptionType import OptionType
from src.Utils.Valuator.Black76 import Black76, Black76Vec

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestPolynomialExtrapolator(TestCase):
    def test_first_order(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # numerical implication of first derivative against analytical
        # expect to see regression with high precision as central difference is applied
        xs = np.linspace(10.0, 200.0, num=191)
        dx = 1e-6
        lam = 0.5
        betas = np.array([1.0, -2.0, 3.0, -4.0]) * 1.0e-6

        for is_reciprocal in [False, True]:
            xs_d = xs * (1.0 - dx)
            xs_u = xs * (1.0 + dx)
            dxs = np.subtract(xs_u, xs_d)
            v_d = PolynomialExtrapolator.zeroth_order(xs_d, betas, lam, is_reciprocal)
            v_u = PolynomialExtrapolator.zeroth_order(xs_u, betas, lam, is_reciprocal)
            first_n = (v_u - v_d) / dxs
            first = PolynomialExtrapolator.first_order(xs, betas, lam, is_reciprocal)
            for idx in range(xs.__len__()):
                rel_diff = first[idx] / first_n[idx] - 1.0
                self.assertAlmostEqual(rel_diff, 0.0, places=6)
                # print('Is reciprocal: ', is_reciprocal, '\nAnalytical', first, '\nNumerical', first_n)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_second_order(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # numerical implication of second derivative against analytical
        # expect to see regression with moderate precision as this is second derivative
        xs = np.linspace(10.0, 200.0, num=191)
        dx = 1e-4
        lam = 0.5
        betas = np.array([1.0, -2.0, 3.0, -4.0]) * 1e-6

        for is_reciprocal in [False, True]:
            xs_d = xs * (1.0 - dx)
            xs_u = xs * (1.0 + dx)
            dxs = np.subtract(xs_u, xs_d) / 2.0
            v_d = PolynomialExtrapolator.zeroth_order(xs_d, betas, lam, is_reciprocal)
            v = PolynomialExtrapolator.zeroth_order(xs, betas, lam, is_reciprocal)
            v_u = PolynomialExtrapolator.zeroth_order(xs_u, betas, lam, is_reciprocal)
            second_n = (v_u - 2.0 * v + v_d) / (dxs ** 2)
            second = PolynomialExtrapolator.second_order(xs, betas, lam, is_reciprocal)
            for idx in range(xs.__len__()):
                rel_diff = second[idx] / second_n[idx] - 1.0
                self.assertAlmostEqual(rel_diff, 0.0, places=4)
                # print('Is reciprocal: ', is_reciprocal, '\nAnalytical', second, '\nNumerical', second_n)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_eta(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # reciprocal recovery, identical function but one is the reciprocal
        # expect to recover derivatives after chain rule
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
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_quadratic_fit(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # fit to the calibration point price, delta_k and gamma_k
        # expect to recover derivatives with high precision
        f = 150.0
        tau = 0.75
        sig = 0.4
        b = 0.96

        for lam, opt_type in [(2.0, OptionType.put), (0.5, OptionType.call)]:
            # for lam, opt_type in [(0.5, OptionType.call)]:
            k = f / lam  # strikes should be on the wings
            price = Black76.price(f, k, tau, sig, b, opt_type)
            delta_k = Black76.delta_k(f, k, tau, sig, b, opt_type)
            gamma_k = Black76.gamma_k(f, k, tau, sig, b)

            pe = PolynomialExtrapolator.quadratic_fit(lam, k, price, delta_k, gamma_k, opt_type)
            is_reciprocal = opt_type == OptionType.call
            zeroth = pe.zeroth_order(k, pe.betas, lam, is_reciprocal)
            first = pe.first_order(k, pe.betas, lam, is_reciprocal)
            second = pe.second_order(k, pe.betas, lam, is_reciprocal)
            self.assertAlmostEqual(price, zeroth, places=12)
            self.assertAlmostEqual(delta_k, first, places=12)
            self.assertAlmostEqual(gamma_k, second, places=12)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_extrapolate(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        # visualize extrapolation implied vol and density function
        # expect to see density function varies with lambda

        def __calc_derivatives(sabr_model: SABRModelLognormalApprox, forward: float, strike: float, ttm: float,
                               opt_type: OptionType, dk: float = 1e-2):
            strike_m = strike * (1.0 - dk)
            strike_p = strike * (1.0 + dk)
            vol_m = sabr_model.calc_vol(forward, strike_m)
            vol = sabr_model.calc_vol(forward, strike)
            vol_p = sabr_model.calc_vol(forward, strike_p)
            # use numerical implication of derivatives here because of SABR model
            price_m = Black76.price(forward, strike_m, ttm, vol_m, 1.0, opt_type)
            price = Black76.price(forward, strike, ttm, vol, 1.0, opt_type)
            price_p = Black76.price(forward, strike_p, ttm, vol_p, 1.0, opt_type)
            delta_k = (price_p - price_m) / (strike_p - strike_m)
            gamma_k = (price_p - 2.0 * price + price_m) / ((strike * dk) ** 2)
            return price, delta_k, gamma_k

        f = 97.4
        tau = 3.5
        b = 1.0
        k_flr, k_left, k_right, k_cap = 1.0, 60.0, 200.0, 300.0
        alpha, beta, nu, rho = 0.12, 1.0, 0.88, -0.66

        ks_full = np.linspace(k_flr, k_cap, num=int(k_cap - k_flr))
        ks_left = np.linspace(k_flr, k_left, num=int(k_left - k_flr))
        ks_right = np.linspace(k_right, k_cap, num=int(k_cap - k_right))

        cases = [(OptionType.put, 1.2), (OptionType.put, 2.0), (OptionType.put, 3.0),
                 (OptionType.call, 2.33), (OptionType.call, 5.0), (OptionType.call, 15.0)]
        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        vols_ful = model.calc_vol_vec(f, ks_full)

        plt.subplot(2, 1, 1)
        plt.title('implied vol')
        plt.plot(ks_full, vols_ful)
        legends = ['SABR']

        for option_type, lam in cases:
            k = k_left if option_type == OptionType.put else k_right
            ks = ks_left if option_type == OptionType.put else ks_right
            zeroth, first, second = __calc_derivatives(model, f, k, tau, option_type)
            pe = PolynomialExtrapolator.quadratic_fit(lam, k, zeroth, first, second, option_type)
            prices = pe.extrapolate(ks)
            vols = Black76Vec.imp_vol(f, ks, tau, prices, b, option_type)
            plt.plot(ks, vols)
            legends += [option_type.name + '_' + lam.__str__()]
        plt.legend(legends)

        plt.subplot(2, 1, 2)
        plt.title('density')
        dens_full, ks_den_ful = model.calc_fwd_den(f, rel_bounds=(k_flr / f, k_cap / f))
        plt.plot(ks_den_ful, dens_full)

        for option_type, lam in cases:
            k = k_left if option_type == OptionType.put else k_right
            ks = ks_left if option_type == OptionType.put else ks_right
            zeroth, first, second = __calc_derivatives(model, f, k, tau, option_type)
            pe = PolynomialExtrapolator.quadratic_fit(lam, k, zeroth, first, second, option_type)
            dens = pe.second_order(ks, pe.betas, lam, option_type == OptionType.call)
            plt.plot(ks, dens)
        plt.legend(legends)
        plt.plot(ks_full, np.zeros(np.shape(ks_full)), 'k--')
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
