from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from src.SABRModel.SABRModel import SABRModel
from src.SABRModel.SABRModelCalibrator import SABRModelCalibratorAlphaNuRho

__author__ = 'frank.ma'


class TestSABRModelCalibrator(TestCase):
    def test_calibrate(self):
        t = 1.5
        alpha = 0.3215
        beta = 1.0
        nu = 0.8632
        rho = -0.2421

        forward = 125.0
        strikes = np.linspace(75.0, 175.0, num=11)

        model = SABRModel(t, alpha, beta, nu, rho)
        vols = model.calc_lognormal_vol_vec_k(forward, strikes)
        weights = np.full(vols.__len__(), 1.0 / vols.__len__())

        init_guess = (1.2, 0.7, -0.4)
        calibrator = SABRModelCalibratorAlphaNuRho(t, forward, strikes, vols, weights, vol_type='Black', beta=beta,
                                                   init_guess=init_guess)
        model_calibrated = calibrator.calibrate()

        print('no perturbation calibration results')
        print('parameter\tinput\tcalibrated\trelative diff')
        print('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_calibrated.alpha, model_calibrated.alpha / alpha - 1.0))
        print('nu\t%.6f\t%.6f\t%.2e' % (nu, model_calibrated.nu, model_calibrated.nu / nu - 1.0))
        print('rho\t%.6f\t%.6f\t%.2e' % (rho, model_calibrated.rho, model_calibrated.rho / rho - 1.0))

        assert abs(alpha / model_calibrated.alpha - 1.0) < 1e-4, 'alpha diff larger than 1e-4'
        assert abs(nu / model_calibrated.nu - 1.0) < 1e-4, 'nu diff larger than 1e-4'
        assert abs(rho / model_calibrated.rho - 1.0) < 1e-4, 'rho diff larger than 1e-4'

        perturbation = np.exp(np.random.normal(loc=0.0, scale=0.01, size=strikes.__len__()))
        vols_pert = vols * perturbation

        calibrator_pert = SABRModelCalibratorAlphaNuRho(t, forward, strikes, vols_pert, weights)
        model_pert_calibrated = calibrator_pert.calibrate()

        vols_pert_res = model_pert_calibrated.calc_lognormal_vol_vec_k(forward, strikes)
        strikes_len = np.linspace(70.0, 180.0, num=101)
        vols_len = model.calc_lognormal_vol_vec_k(forward, strikes_len)
        vols_pert_res_len = model_pert_calibrated.calc_lognormal_vol_vec_k(forward, strikes_len)

        print('with perturbation calibration results')
        print('parameter\tinput\tcalibrated\trelative diff')
        print('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_pert_calibrated.alpha, model_pert_calibrated.alpha / alpha - 1.0))
        print('nu\t%.6f\t%.6f\t%.2e' % (nu, model_pert_calibrated.nu, model_pert_calibrated.nu / nu - 1.0))
        print('rho\t%.6f\t%.6f\t%.2e' % (rho, model_pert_calibrated.rho, model_pert_calibrated.rho / rho - 1.0))

        plt.plot(strikes, vols, 'ro', alpha=0.5)
        plt.plot(strikes_len, vols_len, 'r-', alpha=0.5)
        plt.plot(strikes, vols_pert, 'bo')
        plt.plot(strikes, vols_pert_res, 'bx')
        plt.plot(strikes_len, vols_pert_res_len, 'b-')
        plt.legend(['original', 'original dense', 'perturbed', 'perturbed fitted', 'perturbed fitted dense'])
        plt.xlim([60.0, 190.0])
        plt.ylim([0.30, 0.50])
        plt.show()
