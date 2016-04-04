import logging
from datetime import datetime
import numpy as np

from src.Quote.QuoteVanillaOptionEquity import QuoteVanillaOptionEquity
from src.Utils.Types.PayoffType import PayoffType
from src.VolSurface.ITradableVolSurface import ITradableVolSurface

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class TradableVolSurfaceEquity(ITradableVolSurface):
    def __init__(self, quotes: list):
        # screening the quote set first
        asof = None
        self.opt_eur = []
        self.opt_ame = []
        for idx, quote in enumerate(quotes):
            if not isinstance(quote, QuoteVanillaOptionEquity):
                raise AttributeError('expect object type of %s, received object type of %s at position % i'
                                     % (QuoteVanillaOptionEquity.__name__, quote.__class__.__name__, idx))
            if idx == 0:
                asof = quote.asof  # assume first one is baseline
            else:
                if not asof.__eq__(quote.asof):
                    raise ValueError('expect asof of %s, received misaligned asof of %s at position % i'
                                     % (asof, quote.asof, idx))
            if quote.payoff_type is PayoffType.European:
                self.opt_eur.append(quote)
            elif quote.payoff_type is PayoffType.American:
                self.opt_ame.append(quote)
            else:
                logger.warning('payoff type %s could not be handled, bypass the quote' % quote.payoff_type)

        super().__init__(asof)
        self.quotes = quotes

    def clean(self):
        pass
