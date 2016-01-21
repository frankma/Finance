import logging

import numpy as np
import scipy.optimize as opt

from src.SABRModel.SABRModel import SABRModel, SABRModelLognormalApprox, SABRModelNormalApprox
from src.Utils.VolType import VolType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SABRModelCalibrator(object):
    def __init__(self, alpha_bound: tuple = (0.0001, None), beta_bound: tuple = (-0.001, 1.0001),
                 nu_bound: tuple = (0.0001, None), rho_bound: tuple = (-0.9999, 0.999), tol_lvl_abs: float = 1e-14):
        self.alpha_bound = alpha_bound
        self.beta_bound = beta_bound
        self.nu_bound = nu_bound
        self.rho_bound = rho_bound
        self.tol_lvl_abs = tol_lvl_abs

    def calibrate(self) -> SABRModel:
        pass

    def error_function(self, x: tuple) -> float:
        pass


class SABRModelCalibratorAlphaNuRho(SABRModelCalibrator):
    def __init__(self, t: float, forward: float, strikes: np.array, vols: np.array, weights: np.array,
                 vol_type: VolType, beta: float = 1.0, init_guess: tuple = (0.2, 0.4, -0.25)):
        super().__init__()
        self.t = t
        self.forward = forward
        if strikes.__len__() != vols.__len__() != weights:
            raise ValueError('strikes, vol and weights inputs mismatch in length.')
        if strikes.__len__() < 3:
            raise ValueError('minimum of three quotes need to be passed into calibrator.')
        self.strikes = strikes
        self.vols = vols
        self.vol_type = vol_type
        self.weights = weights
        self.beta = beta
        self.init_guess = init_guess

    def calibrate(self) -> SABRModelNormalApprox or SABRModelLognormalApprox:
        res = opt.minimize(self.error_function, self.init_guess, method='L-BFGS-B', jac=False,
                           bounds=(self.alpha_bound, self.nu_bound, self.rho_bound), tol=self.tol_lvl_abs)
        alpha, nu, rho = res.x
        if self.vol_type == VolType.black:
            return SABRModelLognormalApprox(self.t, alpha, self.beta, nu, rho)
        elif self.vol_type == VolType.normal:
            return SABRModelNormalApprox(self.t, alpha, self.beta, nu, rho)

    def error_function(self, x: tuple) -> float:
        alpha, nu, rho = x

        if self.vol_type == VolType.black:
            model = SABRModelLognormalApprox(self.t, alpha, self.beta, nu, rho)
        elif self.vol_type == VolType.normal:
            model = SABRModelNormalApprox(self.t, alpha, self.beta, nu, rho)
        else:
            raise ValueError('unrecognized volatility type %s' % self.vol_type.__str__())

        imp_vols = model.calc_vol_vec(self.forward, self.strikes)
        errors = np.power((self.vols - imp_vols) * self.weights, 2)
        return sum(errors)


class SABRModelCalibratorNuRho(SABRModelCalibrator):
    def __init__(self, t: float, forward: float, vol_atm: float, strikes: np.array, vols: np.array, weights: np.array,
                 vol_type: VolType, beta: float = 1.0, init_guess: tuple = (0.4, -0.25)):
        super().__init__()
        self.t = t
        self.forward = forward
        self.vol_atm = vol_atm
        if strikes.__len__() != vols.__len__() != weights:
            raise ValueError('strikes, vol and weights inputs mismatch in length.')
        if strikes.__len__() < 3:
            raise ValueError('minimum of three quotes need to be passed into calibrator.')
        self.strikes = strikes
        self.vols = vols
        self.vol_type = vol_type
        self.weights = weights
        self.beta = beta
        self.init_guess = init_guess

    def calibrate(self) -> SABRModelLognormalApprox or SABRModelNormalApprox:
        res = opt.minimize(self.error_function, self.init_guess, method='L-BFGS-B', jac=False,
                           bounds=(self.nu_bound, self.rho_bound), tol=self.tol_lvl_abs)
        nu, rho = res.x
        if self.vol_type == VolType.black:
            alpha = SABRModelLognormalApprox.solve_alpha(self.forward, self.vol_atm, self.t, self.beta, nu, rho)
            return SABRModelLognormalApprox(self.t, alpha, self.beta, nu, rho)
        elif self.vol_type == VolType.normal:
            alpha = SABRModelNormalApprox.solve_alpha(self.forward, self.vol_atm, self.t, self.beta, nu, rho)
            return SABRModelNormalApprox(self.t, alpha, self.beta, nu, rho)
        else:
            raise ValueError('unrecognized volatility type %s' % self.vol_type.__str__())

    def error_function(self, x: tuple):
        nu, rho = x

        if self.vol_type == VolType.black:
            alpha = SABRModelLognormalApprox.solve_alpha(self.forward, self.vol_atm, self.t, self.beta, nu, rho)
            model = SABRModelLognormalApprox(self.t, alpha, self.beta, nu, rho)
        elif self.vol_type == VolType.normal:
            alpha = SABRModelNormalApprox.solve_alpha(self.forward, self.vol_atm, self.t, self.beta, nu, rho)
            model = SABRModelNormalApprox(self.t, alpha, self.beta, nu, rho)
        else:
            raise ValueError('unrecognized volatility type %s' % self.vol_type.__str__())

        imp_vols = model.calc_vol_vec(self.forward, self.strikes)
        errors = np.power((self.vols - imp_vols) * self.weights, 2)
        return sum(errors)
