import datetime
from src.Assets.Option import Option

__author__ = 'frank.ma'


class OptionEuropean(Option):

    def __init__(self, strike: float, expiry: datetime, bid: float, ask: float):
        super().__init__(strike, expiry, bid, ask)