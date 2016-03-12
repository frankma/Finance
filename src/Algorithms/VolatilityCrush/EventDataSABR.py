from datetime import datetime

import numpy as np

from src.SABRModel.SABRMarketData import SABRMarketData
from src.Utils.PayoffType import PayoffType

__author__ = 'frank.ma'


class EventDataSABR(object):
    def __init__(self, asof_pre: datetime, md_pre: dict, asof_post: datetime, md_post: dict, cash_rate: float = 0.0):
        self.asof_pre = asof_pre
        self.md_pre = md_pre
        self.asof_post = asof_post
        self.md_post = md_post
        # minor processing
        expiries_pre = list(md_pre.keys())
        expiries_post = list(md_post.keys())
        self.expiries = list(set(expiries_pre).intersection(expiries_post))  # only find the common expires
        self.cash_rate = cash_rate
        pass

    def get_md_cross_event(self, expiry: datetime):
        if expiry not in self.expiries:
            raise ValueError('expiry (%s) is not cached in market data components.' % expiry)
        mdc_pre = self.md_pre[expiry]
        mdc_post = self.md_post[expiry]
        if not isinstance(mdc_pre, SABRMarketData) and isinstance(mdc_post, SABRMarketData):
            raise ValueError('expect market data components as SABRMarketData')
        return mdc_pre, mdc_post

    def check_cross_instruments(self, expiry: datetime, strikes: np.array, opt_types: np.array,
                                payoff_type: PayoffType):
        md_pre, md_post = self.get_md_cross_event(expiry)
        vols_pre = md_pre.get_vols_from_model(strikes)
        vols_post = md_post.get_vols_from_model(strikes)
        prices_pre = np.empty(strikes.__len__())
        prices_pre.fill(np.nan)
        prices_post = np.empty(strikes.__len__())
        prices_post.fill(np.nan)

        for idx, strike in enumerate(strikes):
            opt_type = opt_types[idx]
            # TODO: finish this
        return prices_pre, prices_post
