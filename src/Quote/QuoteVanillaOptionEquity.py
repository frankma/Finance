import logging
from datetime import datetime

from src.Quote.Quote import Quote
from src.Utils.Types.OptionType import OptionType
from src.Utils.Types.PayoffType import PayoffType

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class QuoteVanillaOptionEquity(Quote):
    def __init__(self, asof: datetime, snap_time: datetime, expiry: datetime, strike: float, opt_type: OptionType,
                 payoff_type: PayoffType, bid: float, ask: float):
        super().__init__(asof, snap_time, bid, ask)
        self.expiry = expiry
        self.strike = strike
        self.opt_type = opt_type
        self.payoff_type = payoff_type
