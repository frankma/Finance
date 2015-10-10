import datetime

from src.Assets.Option import Option

__author__ = 'frank.ma'


class OptionEuropean(Option):
    def __init__(self, identifier: str, expiry: datetime, strike: float):
        super().__init__(identifier, expiry, strike)
