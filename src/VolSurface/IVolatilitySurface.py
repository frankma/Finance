import datetime

__author__ = 'frank.ma'


class IVolatilitySurface(object):
    def __init__(self, asof: datetime):
        self.asof = asof

    def get_volatility(self, expiry: datetime, strike: float):
        pass
