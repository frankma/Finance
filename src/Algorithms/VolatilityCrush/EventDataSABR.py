import numpy as np

from datetime import datetime

from src.SABRModel.SABRMarketData import SABRMarketData
from src.Utils.OptionType import OptionType
from src.Utils.Valuator.Black76 import Black76Vec
from src.Utils.VolType import VolType

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

    def get_mds_cross_event(self, expiry: datetime):
        if expiry not in self.expiries:
            raise ValueError('expiry (%s) is not cached in market data components.' % expiry)
        mdc_pre = self.md_pre[expiry]
        mdc_post = self.md_post[expiry]
        if not isinstance(mdc_pre, SABRMarketData) and isinstance(mdc_post, SABRMarketData):
            raise ValueError('expect market data components as SABRMarketData')
        return mdc_pre, mdc_post

    def check_cross_instruments(self, expiry: datetime, strikes: np.array, opt_types: np.array):
        mdc_pre, mdc_post = self.get_mds_cross_event(expiry)
        prices_pre = []
        prices_post = []
        for idx, strike in enumerate(strikes):
            opt_type = opt_types[idx]
            prices_pre += [mdc_pre.get_price_from_model(strike, opt_type)]
            prices_post += [mdc_post.get_vols_from_model(strike, opt_type)]
        return prices_pre, prices_post
