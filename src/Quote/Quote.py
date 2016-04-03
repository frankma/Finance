import logging
from datetime import datetime

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Quote(object):
    def __init__(self, asof: datetime, snap_time: datetime, bid: float, ask: float):
        self.asof = asof
        self.snap_time = snap_time if snap_time is not None else asof
        self.bid = bid
        self.ask = ask
        self.price = (bid + ask) / 2.0
