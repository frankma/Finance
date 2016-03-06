from datetime import datetime

import numpy as np

from src.SABRModel.SABRModel import SABRModelLognormalApprox, SABRModelNormalApprox
from src.SABRModel.SABRModelCalibrator import SABRModelCalibratorAlphaNuRho
from src.Utils.Valuator.Black76 import Black76
from src.Utils.Valuator.NormalModel import NormalModel
from src.Utils.VolType import VolType

__author__ = 'frank.ma'


class SABRMarketData(object):
    def __init__(self, asof: datetime, expiry: datetime, forward: float, bond: float,
                 tau: float, alpha: float, beta: float, nu: float, rho: float,
                 strikes: np.array, volatilities: np.array, vol_type: VolType):
        self.asof = asof
        self.expiry = expiry
        self.forward = forward
        self.bond = bond
        self.tau = tau
        self.alpha, self.beta, self.nu, self.rho = alpha, beta, nu, rho
        self.strikes = strikes
        self.volatilities = volatilities
        self.vol_type = vol_type
        # some preliminary processings
        self.vol_dict = dict(zip(self.strikes, self.volatilities))
        if self.vol_type == VolType.black:
            self.model = SABRModelLognormalApprox(self.tau, self.alpha, self.beta, self.nu, self.rho)
        elif self.vol_type == VolType.normal:
            self.model = SABRModelNormalApprox(self.tau, self.alpha, self.beta, self.nu, self.rho)
        pass

    @staticmethod
    def calibrate_model_from_vol(asof: datetime, expiry: datetime, forward: float, bond: float,
                                 strikes: np.array, volatilities: np.array, vol_type: VolType,
                                 beta: float = 1.0, initial_guess: tuple = (0.2, 0.4, -0.25)):
        tau = (expiry - asof) / datetime(365.25)  # Act/365.25
        weights = np.full(np.shape(strikes), 1.0 / strikes.__len__())
        calibrator = SABRModelCalibratorAlphaNuRho(tau, forward, strikes, volatilities, weights, vol_type, beta=beta,
                                                   init_guess=initial_guess)
        model = calibrator.calibrate()
        alpha, nu, rho = model.alpha, model.nu, model.rho
        return SABRMarketData(asof, expiry, forward, bond, tau, alpha, beta, nu, rho, strikes, volatilities, vol_type)

    @staticmethod
    def calibrate_model_from_price(asof: datetime, expiry: datetime, forward: float, bond: float,
                                   strikes: np.array, prices: np.array, opt_types: np.array, vol_type: VolType,
                                   beta: float = 1.0, initial_guess: tuple = (0.2, 0.4, -0.25)):
        tau = (expiry - asof) / datetime(365.25)  # Act/365.25
        # first convert volatilities
        volatilities = np.zeros(np.shape(strikes))
        for idx, strike in enumerate(strikes):
            price = prices[idx]
            opt_type = opt_types[idx]
            if vol_type == VolType.black:
                volatilities[idx] = Black76.imp_vol(forward, strike, tau, price, bond, opt_type)
            elif vol_type == VolType.normal:
                volatilities[idx] = NormalModel.imp_vol(forward, strike, tau, price, bond, opt_type)
            else:
                raise NotImplementedError('method not implemented')
        # then calibrate model
        return SABRMarketData.calibrate_model_from_vol(asof, expiry, forward, bond, strikes, volatilities, vol_type,
                                                       beta, initial_guess)

    def get_quote(self, strike: float):
        if strike not in self.strikes:
            raise ValueError('strike %.2f is not in storage' % strike)
        return self.vol_dict[strike]

    def get_quote_from_model(self, strike: float or np.array):
        return self.model.calc_vol_vec(self.forward, strike)


