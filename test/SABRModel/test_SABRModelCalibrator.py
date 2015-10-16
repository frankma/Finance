from unittest import TestCase

import numpy as np

from src.SABRModel.SABRModel import SABRModel
from src.SABRModel.SABRModelCalibrator import SABRModelCalibrator

__author__ = 'frank.ma'


class TestSABRModelCalibrator(TestCase):
    def test_calibrate(self):
        t = 1.5
        alpha = 0.33
        beta = 1.0
        nu = 0.77
        rho = -0.45

        forward = 125.0
        strikes = np.linspace(75.0, 175.0, num=11)

        model = SABRModel(t, alpha, beta, nu, rho)
        vols = model.calc_lognormal_vol_vec_k(forward, strikes)
        weights = np.full(vols.__len__(), 1.0 / vols.__len__())

        init_guess = (1.2, 0.7, -0.4)
        calibrator = SABRModelCalibrator(t, forward, strikes, vols, weights, vol_type='Black', beta=beta,
                                         init_guess=init_guess)
        model_calibrated = calibrator.calibrate()
        print(model_calibrated.alpha, model_calibrated.beta, model_calibrated.nu, model_calibrated.rho)
