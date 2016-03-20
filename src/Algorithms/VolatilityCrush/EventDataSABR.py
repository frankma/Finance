import logging

from datetime import datetime

import numpy as np

from src.SABRModel.SABRMarketData import SABRMarketData
from src.Utils.PayoffType import PayoffType
from src.Utils.Valuator.BAW import BAW
from src.Utils.Valuator.Black76 import Black76

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class EventDataSABR(object):
    def __init__(self, md_pre: list, md_post: list):
        # pre event data sorting and cleaning
        self.md_pre = self.__initial_check(md_pre)
        self.asof_pre = next(iter(self.md_pre.values())).asof
        expiries_pre = list(self.md_pre.keys())
        # post event data sorting and cleaning
        self.md_post = self.__initial_check(md_post)
        self.asof_post = next(iter(self.md_post.values())).asof
        expiries_post = list(self.md_post.keys())
        # find the common expires across event for later reference
        self.expiries = list(set(expiries_pre).intersection(expiries_post))
        pass

    @staticmethod
    def __initial_check(md: list):
        expiries = []
        asof = None
        for idx, item in enumerate(md):
            # restrict instance in the list
            if not isinstance(item, SABRMarketData):
                raise ValueError('item %i in market data pre event is not instance of SABRMarketData' % idx)
            # restrict pre event asof date
            if idx == 0:
                asof = item.asof
            else:
                if asof != item.asof:
                    raise ValueError('unexpected pre event asof date (%s) at position (%i)' % (item.asof, idx))
            # log expiries
            expiries += [item.expiry]
        return dict(zip(expiries, md))

    def get_md_cross_event(self, expiry: datetime):
        if expiry not in self.expiries:
            raise ValueError('expiry (%s) is not cached in market data components.' % expiry)
        md_pre = self.md_pre[expiry]
        md_post = self.md_post[expiry]
        if not isinstance(md_pre, SABRMarketData) and isinstance(md_post, SABRMarketData):
            raise ValueError('expect market data components as SABRMarketData')
        return md_pre, md_post

    @staticmethod
    def calc_prices(md: SABRMarketData, strikes: np.array, opt_types: np.array, payoff_types: np.array):
        spot, fwd, tau = md.spot, md.forward, md.tau
        r = np.log(md.bond) / tau
        q = np.log(md.bond_carry) / tau

        vols = md.get_vols_from_model(strikes)
        prices = np.empty(strikes.__len__())
        prices.fill(np.nan)

        for idx, strike in enumerate(strikes):
            payoff_type = payoff_types[idx]
            if payoff_type is PayoffType.European:
                prices[idx] = BAW.price(spot, strike, tau, r, q, vols[idx], opt_types[idx])
            elif payoff_type is PayoffType.American:
                prices[idx] = Black76.price(fwd, strike, tau, vols[idx], md.bond, opt_types[idx])
            else:
                raise NotImplementedError('in event pricer payoff type (%s) is not implemented yet' % payoff_type)
        return prices

    def calc_prices_cross_event(self, expiry: datetime, strikes: np.array, opt_types: np.array, payoff_types: np.array):
        md_pre, md_post = self.get_md_cross_event(expiry)

        prices_pre = self.calc_prices(md_pre, strikes, opt_types, payoff_types)
        prices_post = self.calc_prices(md_post, strikes, opt_types, payoff_types)

        return prices_pre, prices_post
