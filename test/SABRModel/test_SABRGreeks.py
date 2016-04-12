import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from src.SABRModel.SABRGreeks import SABRGreeks
from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.Interpolator.LinearInterpolator1D import LinearInterpolator1D

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

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pde_k(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

