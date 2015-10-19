import numpy as np
import scipy.optimize as opt

from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'


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
                 vol_type: str = 'black', beta: float = 1.0, init_guess: tuple = (0.2, 0.4, -0.25)):
        super().__init__()
        self.t = t
        self.forward = forward
        if strikes.__len__() != vols.__len__() != weights:
            raise ValueError('strikes, vol and weights inputs mismatch in length.')
        if strikes.__len__() < 3:
            raise ValueError('minimum of three quotes need to be passed into calibrator.')
        self.strikes = strikes
        self.vols = vols
        if vol_type.lower() == 'black':
            self.vol_type = 'black'
        elif vol_type.lower() == 'normal':
            self.vol_type = 'normal'
        else:
            raise ValueError('unrecognized volatility type %s; expect black or normal' % vol_type)
        self.weights = weights
        self.beta = beta
        self.init_guess = init_guess

    def calibrate(self) -> SABRModel:
        res = opt.minimize(self.error_function, self.init_guess, method='L-BFGS-B', jac=False,
                           bounds=(self.alpha_bound, self.nu_bound, self.rho_bound), tol=self.tol_lvl_abs)
        alpha, nu, rho = res.x
        return SABRModel(self.t, alpha, self.beta, nu, rho)

    def error_function(self, x: tuple) -> float:
        alpha, nu, rho = x
        model = SABRModel(self.t, alpha, self.beta, nu, rho)
        if self.vol_type == 'black':
            imp_vols = model.calc_lognormal_vol_vec_k(self.forward, self.strikes)
        else:
            imp_vols = model.calc_normal_vol_vec_k(self.forward, self.strikes)
        errors = np.power((self.vols - imp_vols) * self.weights, 2)
        return sum(errors)
