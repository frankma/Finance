import datetime
import logging

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Quote(object):
    def __init__(self, asof: datetime, bid: float, ask: float):
        self._asof = asof
        self._bid = bid
        self._ask = ask
