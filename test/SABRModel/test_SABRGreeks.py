import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from src.SABRModel.SABRGreeks import SABRGreeks
from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.Interpolator.LinearInterpolator1D import LinearInterpolator1D
from src.Utils.Types.OptionType import OptionType
from src.Utils.Valuator.Black76 import Black76

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestSABRGreeks(TestCase):
    def test_density(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        forward = 150.0
        strikes = np.linspace(50.0, 250.0, num=21)
        tau = 1.75
        b = 1.0
        alpha, beta, nu, rho = 0.2, 1.0, 0.4, -0.33
        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        density, bins = model.calc_fwd_den_sp(forward, rel_bounds=(0.1, 2.0), n_bins=2000)
        interpolator = LinearInterpolator1D(bins, density)
        gamma_k = SABRGreeks.gamma_k(forward, strikes, tau, b, model)
        for kdx, strike in enumerate(strikes):
            den = interpolator.calc(strike)
            logger.debug('%.2f\t%.8f\t%.8f\t%.4e' % (strike, den, gamma_k[kdx], den - gamma_k[kdx]))
            self.assertAlmostEqual(float(den), gamma_k[kdx], places=6)
        plt.plot(strikes, gamma_k, 'x')
        plt.plot(bins, density, '-')
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pde_f(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        forward = 150.0
        strike = 145.0
        tau = 1.75
        b = 1.0
        alpha, beta, nu, rho = 0.2, 1.0, 0.4, -0.33
        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        opt_type = OptionType.call
        sigma = model.calc_vol(forward, strike)
        theta_sabr = SABRGreeks.theta(forward, strike, tau, b, opt_type, model)
        theta_black = Black76.theta(forward, strike, tau, sigma, b, opt_type)
        logger.info('THETA sabr %.6f black %.6f' % (theta_sabr, theta_black))
        delta_sabr = SABRGreeks.delta(forward, strike, tau, b, opt_type, model)
        delta_black = Black76.delta(forward, strike, tau, sigma, b, opt_type)
        logger.info('DELTA sabr %.6f black %.6f' % (delta_sabr, delta_black))
        gamma_sabr = SABRGreeks.gamma(forward, strike, tau, b, model)
        gamma_black = Black76.gamma(forward, strike, tau, sigma, b)
        logger.info('GAMMA sabr %.6f black %.6f' % (gamma_sabr, gamma_black))

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pde_k(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        forward = 150.0
        strike = 145.0
        tau = 1.75
        b = 1.0
        alpha, beta, nu, rho = 0.2, 1.0, 0.4, -0.33
        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        opt_type = OptionType.call
        sigma = model.calc_vol(forward, strike)
        theta_sabr = SABRGreeks.theta(forward, strike, tau, b, opt_type, model)
        theta_black = Black76.theta(forward, strike, tau, sigma, b, opt_type)
        logger.info('THETA sabr %.6f black %.6f' % (theta_sabr, theta_black))
        delta_k_sabr = SABRGreeks.delta_k(forward, strike, tau, b, opt_type, model)
        delta_k_black = Black76.delta_k(forward, strike, tau, sigma, b, opt_type)
        logger.info('DELTA_K sabr %.6f black %.6f' % (delta_k_sabr, delta_k_black))
        gamma_k_sabr = SABRGreeks.gamma_k(forward, strike, tau, b, model)
        gamma_k_black = Black76.gamma_k(forward, strike, tau, sigma, b)
        logger.info('GAMMA_K sabr %.6f black %.6f' % (gamma_k_sabr, gamma_k_black))

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

