import logging

import pandas as pd

from src.Quote.QuoteVanillaOptionEquity import QuoteVanillaOptionEquity
from src.Utils.Types.PayoffType import PayoffType
from src.VolSurface.ITradableVolSurface import ITradableVolSurface

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class TradableVolSurfaceEquity(ITradableVolSurface):
    def __init__(self, quotes: list):
        # screening the quote set first
        asof = None
        payoff_type = None
        self.opt_eur = []
        self.opt_ame = []
        for idx, quote in enumerate(quotes):
            if not isinstance(quote, QuoteVanillaOptionEquity):
                logger.warning('expect object type of %s, received object type of %s at position %i; bypass it'
                               % (QuoteVanillaOptionEquity.__name__, quote.__class__.__name__, idx))
                continue
            if idx == 0:
                # always take the first one as baseline
                asof = quote.asof
                payoff_type = quote.payoff_type
            else:
                if not asof.__eq__(quote.asof):
                    logger.warning('expect asof %s, received asof %s at position %i; bypass it'
                                   % (asof, quote.asof, idx))
                    continue
                if quote.opt_type is not payoff_type:
                    logger.warning('expect payoff type of %s, received payoff type %s at position %i; bypass it'
                                   % (payoff_type.name, quote.payoff_type.name, idx))
                    continue
            q = [quote.asof, quote.expiry, quote.strike, quote.opt_type, quote.bid, quote.ask]
            if quote.payoff_type is PayoffType.European:
                self.opt_eur.append(q)
            elif quote.payoff_type is PayoffType.American:
                self.opt_ame.append(q)
            else:
                logger.warning('payoff type %s could not be handled, bypass the quote' % quote.payoff_type)

        super().__init__(asof)
        self.quotes = quotes

    @staticmethod
    def __create_df(quote_table, columns=None, id_contact=None):
        # give defaults to mutable inputs
        if columns is None:
            columns = ['asof', 'expiry', 'strike', 'opt_type', 'bid', 'ask']
        if id_contact is None:
            id_contact = ['asof', 'expiry', 'strike', 'opt_type']

        # create data frame and drop duplicates
        df = pd.DataFrame(quote_table, columns=columns)
        df['id'] = df.apply(lambda x: '_'.join(x[id_contact].astype(str)), axis=1)
        df = df.drop_duplicates(subset=['id'])
        return df

    def clean(self):
        # firstly European style options
        df_eur = self.__create_df(self.opt_eur)

        # secondly American style options
        df_ame = self.__create_df(self.opt_ame)

        # lastly, if european available, try to imply forward
        if df_eur.__len__() > 2:
            # do something
            pass

        pass
