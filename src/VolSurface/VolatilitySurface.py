import datetime

__author__ = 'frank.ma'


class VolatilitySurface(object):

    def __init__(self, asof: datetime):
        self.asof = asof