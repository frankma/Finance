import logging
from datetime import datetime

from src.VolSurface.IVolatilitySurface import IVolatilitySurface

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class ITradableVolSurface(object):
    def __init__(self, asof: datetime):
        self.asof = asof
        self.clean()

    def clean(self):
        pass

    def model_volatility_surface(self) -> IVolatilitySurface:
        pass

    def get_quote(self, expiry: datetime, strike: float):
        pass

