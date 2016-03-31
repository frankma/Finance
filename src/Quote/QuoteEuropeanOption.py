import logging
from datetime import datetime

from src.Quote.Quote import Quote
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class QuoteEuropeanOption(Quote):
    def __init__(self, asof: datetime, expiry: datetime, strike: float, opt_type: OptionType, bid: float, ask: float):
        super().__init__(asof, bid, ask)
        self.expiry = expiry
        self.strike = strike
        self.opt_type = opt_type
