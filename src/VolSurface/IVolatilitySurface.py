import logging
from datetime import datetime

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class IVolatilitySurface(object):
    def __init__(self, asof: datetime):
        self.asof = asof

    def get_normal_vol(self, expiry: datetime, strike: float):
        pass

    def get_lognormal_vol(self, expiry: datetime, strike: float):
        pass
