import datetime

__author__ = 'frank.ma'


class Option(object):

    def __init__(self, strike: float, expiry: datetime, bid: float, ask: float):
        self.strike = strike
        self.expiry = expiry
        self.bid = bid
        self.ask = ask