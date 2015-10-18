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
        for _ in range(10 ** 4):
            model.calc_lognormal_vol_vec_k(forward, strikes)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_lognormal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
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
                                                     ' forward %.2f with diff %.14f.' % (forward, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_lognormal_vol_vec_f(forwards, strike)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_lognormal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_normal_vol_vec_k(self):
        tau = 0.25
        alpha = 0.2
        beta = 0.0
        nu = 0.4
        rho = -0.25

        model = SABRModel(tau, alpha, beta, nu, rho)

        forward = 0.05
        strikes = np.linspace(-0.1, 0.2, num=21)

        norm_vols = model.calc_normal_vol_vec_k(forward, strikes)

        for idx, strike in enumerate(strikes):
            vol = model.calc_normal_vol(forward, strike)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                      ' strike %.2f with diff %.14f' % (strike, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_normal_vol_vec_k(forward, strikes)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_normal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_normal_vol_vec_f(self):
        tau = 0.25
        alpha = 0.2
        beta = 0.0
        nu = 0.4
        rho = -0.25

        model = SABRModel(tau, alpha, beta, nu, rho)

        forwards = np.linspace(-0.1, 0.2, num=21)
        strike = 0.05

        norm_vols = model.calc_normal_vol_vec_f(forwards, strike)

        for idx, forward in enumerate(forwards):
            vol = model.calc_normal_vol(forward, strike)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                      ' forward %.2f with diff %.14f' % (forward, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_normal_vol_vec_f(forwards, strike)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_normal_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_solve_alpha_from_atm_vol_given_nu_and_rho(self):
        tau = 0.25
        alpha = 0.2
        beta = 1.0
        nu = 0.4
        rho = -0.25

        model = SABRModel(tau, alpha, beta, nu, rho)

        forward = 150.0

        black_vol = model.calc_lognormal_vol(forward, forward)
        alpha_solved = model.solve_alpha_from_atm_vol_given_nu_and_rho(forward, black_vol, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('lognormal model\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e' % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-2, 'solved alpha differs from input larger than one percent.'

        beta = 0.0
        forward = 0.05
        model_norm = SABRModel(tau, alpha, beta, nu, rho)
        black_vol = model_norm.calc_lognormal_vol(forward, forward)
        alpha_solved = model_norm.solve_alpha_from_atm_vol_given_nu_and_rho(forward, black_vol, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('normal model\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e' % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'
        pass
