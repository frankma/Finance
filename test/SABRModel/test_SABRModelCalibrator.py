import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from src.SABRModel.SABRModelCalibrator import SABRModelCalibratorAlphaNuRho, SABRModelCalibratorNuRho
from src.SABRModel.SABRModelLognormalApprox import SABRModelLognormalApprox
from src.Utils.Types.VolType import VolType

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestSABRModelCalibrator(TestCase):
    def test_calibrate(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        t = 1.5
        alpha = 0.3215
        beta = 1.0
        nu = 0.8632
        rho = -0.2421

        forward = 125.0
        strikes = np.linspace(75.0, 175.0, num=11)

        model = SABRModelLognormalApprox(t, alpha, beta, nu, rho)
        vols = model.calc_vol_vec(forward, strikes)
        weights = np.full(vols.__len__(), 1.0 / vols.__len__())

        init_guess = (1.2, 0.7, -0.4)
        calibrator = SABRModelCalibratorAlphaNuRho(t, forward, strikes, vols, weights, vol_type=VolType.black,
                                                   beta=beta, init_guess=init_guess)
        model_cal = calibrator.calibrate()

        logger.info('no perturbation calibration results')
        logger.info('parameter\tinput\tcalibrated\trelative diff')
        logger.info('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_cal.alpha, model_cal.alpha / alpha - 1.0))
        logger.info('nu\t%.6f\t%.6f\t%.2e' % (nu, model_cal.nu, model_cal.nu / nu - 1.0))
        logger.info('rho\t%.6f\t%.6f\t%.2e' % (rho, model_cal.rho, model_cal.rho / rho - 1.0))

        self.assertLess(abs(alpha / model_cal.alpha - 1.0), 1e-4, msg='alpha diff larger than 1e-4')
        self.assertLess(abs(nu / model_cal.nu - 1.0), 1e-4, msg='nu diff larger than 1e-4')
        self.assertLess(abs(rho / model_cal.rho - 1.0), 1e-4, msg='rho diff larger than 1e-4')

        perturbation = np.exp(np.random.normal(loc=0.0, scale=0.01, size=strikes.__len__()))
        vols_pert = vols * perturbation

        calibrator_pert = SABRModelCalibratorAlphaNuRho(t, forward, strikes, vols_pert, weights, vol_type=VolType.black)
        model_pert_cal = calibrator_pert.calibrate()

        vols_pert_res = model_pert_cal.calc_vol_vec(forward, strikes)
        strikes_len = np.linspace(70.0, 180.0, num=101)
        vols_len = model.calc_vol_vec(forward, strikes_len)
        vols_pert_res_len = model_pert_cal.calc_vol_vec(forward, strikes_len)

        logger.info('with perturbation calibration results')
        logger.info('parameter\tinput\tcalibrated\trelative diff')
        logger.info('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_pert_cal.alpha, model_pert_cal.alpha / alpha - 1.0))
        logger.info('nu\t%.6f\t%.6f\t%.2e' % (nu, model_pert_cal.nu, model_pert_cal.nu / nu - 1.0))
        logger.info('rho\t%.6f\t%.6f\t%.2e' % (rho, model_pert_cal.rho, model_pert_cal.rho / rho - 1.0))

        plt.plot(strikes, vols, 'ro', alpha=0.5)
        plt.plot(strikes_len, vols_len, 'r-', alpha=0.5)
        plt.plot(strikes, vols_pert, 'bo')
        plt.plot(strikes, vols_pert_res, 'bx')
        plt.plot(strikes_len, vols_pert_res_len, 'b-')
        plt.legend(['original', 'original dense', 'perturbed', 'perturbed fitted', 'perturbed fitted dense'])
        plt.xlim([60.0, 190.0])
        plt.ylim([0.30, 0.50])
        plt.show()

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_calibrate_fit_alpha(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        t = 1.5
        alpha = 0.3215
        beta = 1.0
        nu = 0.8632
        rho = -0.2421

        forward = 125.0
        strikes = np.linspace(75.0, 175.0, num=11)

        model = SABRModelLognormalApprox(t, alpha, beta, nu, rho)
        vols = model.calc_vol_vec(forward, strikes)
        vol_atm = model.calc_vol(forward, forward)
        weights = np.full(vols.__len__(), 1.0 / vols.__len__())

        init_guess = (0.7, -0.4)
        calibrator = SABRModelCalibratorNuRho(t, forward, vol_atm, strikes, vols, weights, vol_type=VolType.black,
                                              beta=beta, init_guess=init_guess)
        model_cal = calibrator.calibrate()

        logger.info('no perturbation calibration results')
        logger.info('parameter\tinput\tcalibrated\trelative diff')
        logger.info('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_cal.alpha, model_cal.alpha / alpha - 1.0))
        logger.info('nu\t%.6f\t%.6f\t%.2e' % (nu, model_cal.nu, model_cal.nu / nu - 1.0))
        logger.info('rho\t%.6f\t%.6f\t%.2e' % (rho, model_cal.rho, model_cal.rho / rho - 1.0))

        self.assertLess(abs(alpha / model_cal.alpha - 1.0), 1e-4, msg='alpha diff larger than 1e-4')
        self.assertLess(abs(nu / model_cal.nu - 1.0), 1e-4, msg='nu diff larger than 1e-4')
        self.assertLess(abs(rho / model_cal.rho - 1.0), 1e-4, msg='rho diff larger than 1e-4')

        # give a higher perturbation to check if ATM is aligned
        perturbation = np.exp(np.random.normal(loc=0.0, scale=0.05, size=strikes.__len__()))
        vols_pert = vols * perturbation

        calibrator_pert = SABRModelCalibratorNuRho(t, forward, vol_atm, strikes, vols_pert, weights,
                                                   vol_type=VolType.black, beta=beta, init_guess=init_guess)
        model_pert_cal = calibrator_pert.calibrate()

        vols_pert_res = model_pert_cal.calc_vol_vec(forward, strikes)
        strikes_len = np.linspace(70.0, 180.0, num=101)
        vols_len = model.calc_vol_vec(forward, strikes_len)
        vols_pert_res_len = model_pert_cal.calc_vol_vec(forward, strikes_len)

        logger.info('with perturbation calibration results')
        logger.info('parameter\tinput\tcalibrated\trelative diff')
        logger.info('alpha\t%.6f\t%.6f\t%.2e' % (alpha, model_pert_cal.alpha, model_pert_cal.alpha / alpha - 1.0))
        logger.info('nu\t%.6f\t%.6f\t%.2e' % (nu, model_pert_cal.nu, model_pert_cal.nu / nu - 1.0))
        logger.info('rho\t%.6f\t%.6f\t%.2e' % (rho, model_pert_cal.rho, model_pert_cal.rho / rho - 1.0))

        plt.plot(strikes, vols, 'ro', alpha=0.5)
        plt.plot(strikes_len, vols_len, 'r-', alpha=0.5)
        plt.plot(strikes, vols_pert, 'bo')
        plt.plot(strikes, vols_pert_res, 'bx')
        plt.plot(strikes_len, vols_pert_res_len, 'b-')
        plt.legend(['original', 'original dense', 'perturbed', 'perturbed fitted', 'perturbed fitted dense'])
        plt.xlim([60.0, 190.0])
        plt.ylim([0.30, 0.50])
        plt.show()

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
