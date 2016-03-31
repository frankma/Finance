from datetime import datetime
import numpy as np
from src.VolSurface.IVolatilitySurface import IVolatilitySurface

__author__ = 'frank.ma'


class VolatilitySurfaceEquity(IVolatilitySurface):
    def __init__(self, asof: datetime, expiries: np.array, strikes: np.array, volatilities: np.array):
        super().__init__(asof)
        self.expiries = expiries
        self.strikes = strikes
        self.volatilities = volatilities

