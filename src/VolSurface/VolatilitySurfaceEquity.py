import logging
from datetime import datetime, timedelta

import numpy as np

from src.SABRModel.SABRModel import SABRModel
from src.SABRModel.SABRModelCalibrator import SABRModelCalibratorAlphaNuRho
from src.Utils.VolType import VolType
from src.VolSurface.IVolatilitySurface import IVolatilitySurface

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class VolatilitySurfaceEquity(IVolatilitySurface):
    def __init__(self, asof: datetime, expiries: np.array, forwards: np.array, strikes: np.array,
                 volatilities: np.array, vol_type: VolType):
        super().__init__(asof)
        self.expiries = expiries
        self.strikes = strikes
        self.forwards = dict(zip(expiries, forwards))
        self.volatilities = volatilities
        self.vol_type = vol_type


class VolatilitySurfaceEquitySABR(VolatilitySurfaceEquity):
    def __init__(self, asof: datetime, expiries: np.array, forwards: np.array, strikes: np.array,
                 volatilities: np.array, vol_type: VolType, beta: float = 1.0):
        super().__init__(asof, expiries, forwards, strikes, volatilities, vol_type)
        self.models = dict()
        for idx, expiry in enumerate(expiries):
            fwd = forwards[idx]
            ks = strikes[idx]
            vs = volatilities[idx]
            tau = (expiry - asof) / timedelta(365.25)
            weights = np.full(np.shape(ks), 1.0)  # equal weights as input is volatility
            # TODO: there should be a try and catch process around the model calibration
            model = SABRModelCalibratorAlphaNuRho(tau, fwd, ks, vs, weights, vol_type, beta).calibrate()
            self.models.update({expiry, model})

    def get_vol(self, expiry: datetime, strike: float or np.array, vol_type: VolType):
        if expiry in self.expiries:
            # on node case, get the values directly
            fwd = self.forwards[expiry]
            model = self.models[expiry]
        else:
            # TODO: use interpolation here
            raise NotImplementedError()
        if not isinstance(model, SABRModel):
            raise KeyError('expect the model to be SABR model')

        return model.calc_vol(fwd, strike, vol_type)
