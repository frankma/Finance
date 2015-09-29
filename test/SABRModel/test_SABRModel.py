from unittest import TestCase
import time as tm

import numpy as np

from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'


class TestSABRModel(TestCase):

    def test_calc_lognormal_vol_vec_k(self):
        tau = 0.25
        alpha = 0.04
        beta = 0.5
        nu = 0.4
        rho = -0.45

        model = SABRModel(tau, alpha, beta, nu, rho)

        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=21)
        # precision test
        imp_vols = model.calc_lognormal_vol_vec_k(forward, strikes)
        for idx, strike in enumerate(strikes):
            vol = model.calc_lognormal_vol(forward, strike)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                                     ' strike %.2f with diff %.14f.' % (strike, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10**4):
            model.calc_lognormal_vol_vec_k(forward, strikes)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10**4):
            for strike in strikes:
                model.calc_normal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized %6f, plain %6f, diff %6f' % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_lognormal_vol_vec_f(self):
        tau = 0.25
        alpha = 0.04
        beta = 0.5
        nu = 0.4
        rho = -0.45

        model = SABRModel(tau, alpha, beta, nu, rho)

        strike = 150.0
        forwards = np.linspace(100.0, 200.0, num=21)

        imp_vols = model.calc_lognormal_vol_vec_f(forwards, strike)

        for idx, forward in enumerate(forwards):
            vol = model.calc_lognormal_vol(forward, strike)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                                     ' strike %.2f with diff %.14f.' % (strike, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10**4):
            model.calc_lognormal_vol_vec_f(forwards, strike)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10**4):
            for forward in forwards:
                model.calc_normal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized %6f, plain %6f, diff %6f' % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass