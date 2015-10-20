from unittest import TestCase
import time as tm

import numpy as np

from src.SABRModel.SABRModel import SABRModel, SABRModelLognormalApprox, SABRModelNormalApprox
from src.Utils.VolType import VolType

__author__ = 'frank.ma'


class TestSABRModel(TestCase):
    def test_calc_vol_vec_k_lognormal(self):
        tau = 0.25
        alpha = 0.04
        beta = 0.5
        nu = 0.4
        rho = -0.45

        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)

        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=21)
        # precision test
        imp_vols = model.calc_vol_vec_k(forward, strikes)
        for idx, strike in enumerate(strikes):
            vol = model.calc_vol(forward, strike)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                                     ' strike %.2f with diff %.14f.' % (strike, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec_k(forward, strikes)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_f_lognormal(self):
        tau = 0.25
        alpha = 0.04
        beta = 0.5
        nu = 0.4
        rho = -0.45

        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)

        strike = 150.0
        forwards = np.linspace(100.0, 200.0, num=21)

        imp_vols = model.calc_vol_vec_f(forwards, strike)

        for idx, forward in enumerate(forwards):
            vol = model.calc_vol(forward, strike)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                                     ' forward %.2f with diff %.14f.' % (forward, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec_f(forwards, strike)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_k_normal(self):
        tau = 0.25
        alpha = 0.2
        beta = 0.0
        nu = 0.4
        rho = -0.25

        model = SABRModelNormalApprox(tau, alpha, beta, nu, rho)

        forward = 0.05
        strikes = np.linspace(-0.1, 0.2, num=21)

        norm_vols = model.calc_vol_vec_k(forward, strikes)

        for idx, strike in enumerate(strikes):
            vol = model.calc_vol(forward, strike)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                      ' strike %.2f with diff %.14f' % (strike, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec_k(forward, strikes)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_f_normal(self):
        tau = 0.25
        alpha = 0.2
        beta = 0.0
        nu = 0.4
        rho = -0.25

        model = SABRModelNormalApprox(tau, alpha, beta, nu, rho)

        forwards = np.linspace(-0.1, 0.2, num=21)
        strike = 0.05

        norm_vols = model.calc_vol_vec_f(forwards, strike)

        for idx, forward in enumerate(forwards):
            vol = model.calc_vol(forward, strike)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                      ' forward %.2f with diff %.14f' % (forward, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec_f(forwards, strike)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_vol(forward, strike)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_solve_alpha_from_atm_vol(self):
        tau = 0.25
        alpha = 0.2234
        beta = 1.0
        nu = 0.4
        rho = -0.25

        forward = 150.0

        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        black_vol = model_lognormal.calc_vol(forward, forward)
        alpha_solved = SABRModel.solve_alpha(forward, black_vol, tau, beta, nu, rho, vol_type=VolType.black)
        rel_diff = alpha_solved / alpha - 1.0
        print('lognormal model black vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)
        norm_vol = model_normal.calc_vol(forward, forward)
        alpha_solved = SABRModel.solve_alpha(forward, norm_vol, tau, beta, nu, rho, vol_type=VolType.normal)
        rel_diff = alpha_solved / alpha - 1.0
        print('lognormal model normal vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        beta = 0.0
        forward = 0.05
        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        black_vol = model_lognormal.calc_vol(forward, forward)
        alpha_solved = SABRModel.solve_alpha(forward, black_vol, tau, beta, nu, rho, vol_type=VolType.black)
        rel_diff = alpha_solved / alpha - 1.0
        print('normal model black vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)
        norm_vol = model_normal.calc_vol(forward, forward)
        alpha_solved = SABRModel.solve_alpha(forward, norm_vol, tau, beta, nu, rho, vol_type=VolType.normal)
        rel_diff = alpha_solved / alpha - 1.0
        print('normal model normal vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        pass
