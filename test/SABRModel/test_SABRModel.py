import time as tm
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
        taus = np.linspace(0.01, 5.01, num=31)
        alpha, beta, nu, rho = 0.22, 1.0, 0.55, -0.33
        forward = 150.0
        mu = 0.01
        strikes = np.linspace(100.0, 200.0, num=51)

        blk_vols = np.zeros((taus.__len__(), strikes.__len__()))
        loc_vols = np.zeros((taus.__len__(), strikes.__len__()))

        for tdx, tau in enumerate(taus):
            model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
            blk_vols[tdx] = model.calc_vol_vec(forward, strikes)
            loc_vols[tdx] = model.calc_loc_vol_vec(forward, strikes, mu)

        kk, tt = np.meshgrid(strikes, taus)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_wireframe(kk, tt, blk_vols, color='b', alpha=0.75)
        ax.plot_wireframe(kk, tt, loc_vols, color='r', alpha=0.75)
        ax.legend(['black vol', 'local vol'])
        plt.show()

        pass

    def test_back_bone(self):
        # source: F. Rouah "The SABR Model"
        t = 1.0
        alpha, beta, nu, rho = 0.0104364434616, 0.0, 0.569567801708, 0.260128643732
        model_nrm = SABRModelLognormalApprox(t, alpha, beta, nu, rho)

        alpha, beta, nu, rho = 0.13927, 1.0, 0.57788, -0.06867
        model_lgn = SABRModelLognormalApprox(t, alpha, beta, nu, rho)

        forwards = [0.065, 0.076, 0.088]
        strikes = np.linspace(0.04, 0.11, num=71)

        plt.subplot(2, 1, 1)
        legends = []
        for forward in forwards:
            vols = model_nrm.calc_vol_vec(forward, strikes)
            plt.plot(strikes, vols)
            legends.append('f = %.3f' % forward)
        bb = np.ones(strikes.__len__())
        for idx, strike in enumerate(strikes):
            bb[idx] = model_nrm.calc_vol(strike, strike)
        plt.plot(strikes, bb, '--')
        legends.append('Backbone')
        plt.legend(legends)
        plt.ylim([0.08, 0.22])

        plt.subplot(2, 1, 2)
        legends = []
        for forward in forwards:
            vols = model_lgn.calc_vol_vec(forward, strikes)
            plt.plot(strikes, vols)
            legends.append('f = %.3f' % forward)
        bb = np.ones(strikes.__len__())
        for idx, strike in enumerate(strikes):
            bb[idx] = model_lgn.calc_vol(strike, strike)
        plt.plot(strikes, bb, '--')
        legends.append('Backbone')
        plt.legend(legends)
        plt.ylim([0.08, 0.22])

        plt.show()
        pass

    def test_calc_fwd_den_sp(self):
        t = 3.0
        alpha, beta, nu, rho = 0.2, 1.0, 0.4, -0.25
        model = SABRModelLognormalApprox(t, alpha, beta, nu, rho)

        forward = 150.0
        den_sim, bins_sim = model.sim_fwd_den(forward)
        den_num, bins_num = model.calc_fwd_den(forward)
        den_sp, bins_sp = model.calc_fwd_den_sp(forward)

        plt.plot(bins_sim, den_sim)
        plt.plot(bins_num, den_num)
        plt.plot(bins_sp, den_sp)
        plt.legend(['simulated', 'numerical', 'analytical'])
        plt.xlim([0.0, 2.5 * forward])
        plt.title('SABR model density functions')
        plt.show()

        pass
