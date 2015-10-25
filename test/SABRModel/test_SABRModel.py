from unittest import TestCase
import time as tm

import numpy as np
import matplotlib.pyplot as plt

from src.SABRModel.SABRModel import SABRModelLognormalApprox, SABRModelNormalApprox
from src.Utils.VolType import VolType

__author__ = 'frank.ma'


class TestSABRModel(TestCase):
    def test_calc_vol_vec_k_lognormal(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.04, 0.5, 0.4, -0.45

        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)

        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=21)
        # precision test
        imp_vols = model.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        for idx, strike in enumerate(strikes):
            vol = model.calc_vol(forward, strike, vol_type=VolType.black)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                                     ' strike %.2f with diff %.14f.' % (strike, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec(forward, strikes, vol_type=VolType.black)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_vol(forward, strike, vol_type=VolType.black)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_f_lognormal(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.04, 0.5, 0.4, -0.45

        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)

        strike = 150.0
        forwards = np.linspace(100.0, 200.0, num=21)

        imp_vols = model.calc_vol_vec(forwards, strike, vol_type=VolType.black)

        for idx, forward in enumerate(forwards):
            vol = model.calc_vol(forward, strike, vol_type=VolType.black)
            assert abs(vol - imp_vols[idx]) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                                     ' forward %.2f with diff %.14f.' % (forward, (vol - imp_vols[idx]))
        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec(forwards, strike, vol_type=VolType.black)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_vol(forward, strike, vol_type=VolType.black)
        toc_sca = tm.time() - tic
        print('Calculation time. vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_k_normal(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.2, 0.0, 0.4, -0.25

        model = SABRModelNormalApprox(tau, alpha, beta, nu, rho)

        forward = 0.05
        strikes = np.linspace(-0.1, 0.2, num=31)

        norm_vols = model.calc_vol_vec(forward, strikes, vol_type=VolType.normal)

        for idx, strike in enumerate(strikes):
            vol = model.calc_vol(forward, strike, vol_type=VolType.normal)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'strike vectorization result differs from scalar calculation on' \
                                      ' strike %.2f with diff %.14f' % (strike, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec(forward, strikes, vol_type=VolType.normal)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for strike in strikes:
                model.calc_vol(forward, strike, vol_type=VolType.normal)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized strike %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_calc_vol_vec_f_normal(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.2, 0.0, 0.4, -0.25

        model = SABRModelNormalApprox(tau, alpha, beta, nu, rho)

        forwards = np.linspace(-0.1, 0.2, num=31)
        strike = 0.05

        norm_vols = model.calc_vol_vec(forwards, strike, vol_type=VolType.normal)

        for idx, forward in enumerate(forwards):
            vol = model.calc_vol(forward, strike, vol_type=VolType.normal)
            diff = vol - norm_vols[idx]
            assert abs(diff) < 1e-12, 'forward vectorization result differs from scalar calculation on' \
                                      ' forward %.2f with diff %.14f' % (forward, diff)

        # speed test
        tic = tm.time()
        for _ in range(10 ** 4):
            model.calc_vol_vec(forwards, strike, vol_type=VolType.normal)
        toc_vec = tm.time() - tic
        tic = tm.time()
        for _ in range(10 ** 4):
            for forward in forwards:
                model.calc_vol(forward, strike, vol_type=VolType.normal)
        toc_sca = tm.time() - tic
        print('Calculation time, vectorized forward %.6f, scalar %.6f, diff %.6f'
              % (toc_vec, toc_sca, (toc_vec - toc_sca)))
        pass

    def test_solve_alpha(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.2234, 1.0, 0.4, -0.25

        forward = 150.0

        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        black_vol = model_lognormal.calc_vol(forward, forward)
        alpha_solved = SABRModelLognormalApprox.solve_alpha(forward, black_vol, tau, beta, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('lognormal model black vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)
        norm_vol = model_normal.calc_vol(forward, forward)
        alpha_solved = SABRModelNormalApprox.solve_alpha(forward, norm_vol, tau, beta, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('lognormal model normal vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        beta = 0.0
        forward = 0.05
        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        black_vol = model_lognormal.calc_vol(forward, forward)
        alpha_solved = SABRModelLognormalApprox.solve_alpha(forward, black_vol, tau, beta, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('normal model black vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)
        norm_vol = model_normal.calc_vol(forward, forward)
        alpha_solved = SABRModelNormalApprox.solve_alpha(forward, norm_vol, tau, beta, nu, rho)
        rel_diff = alpha_solved / alpha - 1.0
        print('normal model normal vol\n input\t solved \t rel diff\n %.6f\t%.6f\t%.4e'
              % (alpha, alpha_solved, rel_diff))
        assert abs(rel_diff) < 1e-12, 'solved alpha differs from input larger than one percent.'

        pass

    def test_get_model_lognormal_approx(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.22, 1.0, 0.55, -0.33
        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=11)

        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)
        model_normal_trans_lognormal = model_normal.get_model_lognormal_approx()

        assert isinstance(model_normal_trans_lognormal, SABRModelLognormalApprox)

        for strike in strikes:
            vol_lognormal = model_lognormal.calc_vol(forward, strike)
            vol_lognormal_trans = model_normal_trans_lognormal.calc_vol(forward, strike)
            assert abs(vol_lognormal - vol_lognormal_trans) < 1e-12, 'transformed model mismatch'

        pass

    def test_get_model_normal_approx(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.22, 1.0, 0.55, -0.33
        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=11)

        model_lognormal = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        model_lognormal_trans_normal = model_lognormal.get_model_normal_approx()
        model_normal = SABRModelNormalApprox(tau, alpha, beta, nu, rho)

        assert isinstance(model_lognormal_trans_normal, SABRModelNormalApprox)

        for strike in strikes:
            vol_normal = model_normal.calc_vol(forward, strike)
            vol_normal_trans = model_lognormal_trans_normal.calc_vol(forward, strike)
            assert abs(vol_normal - vol_normal_trans) < 1e-12, 'transformed model mismatch'

        pass

    def test_calc_loc_vol_vec(self):
        tau = 0.25
        alpha, beta, nu, rho = 0.22, 1.0, 0.55, -0.33
        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=11)

        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        blk_vols = model.calc_vol_vec(forward, strikes)
        loc_vols = model.calc_loc_vol_vec(forward, strikes, 0.01)

        plt.plot(strikes, blk_vols)
        plt.plot(strikes, loc_vols)
        plt.legend(['black', 'local'])
        plt.show()
