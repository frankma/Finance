import logging
from datetime import datetime

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class ISDAYieldCurve(object):
    def __init__(self, asof: datetime, tenors: list, rates: list):
        self.asof = asof
        self.tenors = tenors
        self.rates = rates
        pass

    def calc_df(self, date: datetime):
        if date < self.asof:
            raise ValueError('request time %s is before asof date %s' % (date, self.asof))

        pass
