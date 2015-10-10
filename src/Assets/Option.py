import datetime

from src.Assets.Asset import Asset

__author__ = 'frank.ma'


class Option(Asset):
    def __init__(self, identifier: str, expiry: datetime, strike: float):
        super().__init__(identifier)
        self.expiry = expiry
        self.strike = strike
