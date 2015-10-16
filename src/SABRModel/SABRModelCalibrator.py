import numpy as np
import scipy.optimize as opt

from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'


class SABRModelCalibrator(object):
    ALPHA_BOUND = (0.0001, None)
    NU_BOUND = (0.0001, None)
    RHO_BOUND = (-0.9999, 0.9999)
    TOL_LEVEL = 1e-14

    def __init__(self, t: float, forward: float, strikes: np.array, vols: np.array, weights: np.array,
                 vol_type: str = 'black', beta: float = 1.0, init_guess: tuple = (0.2, 0.4, -0.25)):
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
        res = opt.minimize(self.target_function_alpha_nu_rho, self.init_guess, method='L-BFGS-B', jac=False,
                           bounds=(self.ALPHA_BOUND, self.NU_BOUND, self.RHO_BOUND), tol=self.TOL_LEVEL)
        alpha, nu, rho = res.x
        return SABRModel(self.t, alpha, self.beta, nu, rho)

    def target_function_alpha_nu_rho(self, x: tuple) -> float:
        alpha, nu, rho = x
        model = SABRModel(self.t, alpha, self.beta, nu, rho)
        if self.vol_type == 'black':
            imp_vols = model.calc_lognormal_vol_vec_k(self.forward, self.strikes)
        else:
            imp_vols = model.calc_normal_vol_vec_k(self.forward, self.strikes)
        errors = np.power((self.vols - imp_vols) * self.weights, 2)
        return sum(errors)
