import datetime
import logging

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class IVolatilitySurface(object):
    def __init__(self, asof: datetime):
        self.asof = asof

    def get_volatility(self, expiry: datetime, strike: float):
        pass
