import logging
import numpy as np

from src.Utils.Valuator.CashFlowDiscounter import CashFlowDiscounter

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class MarketRateToZeroRate(object):
    def __init__(self, bonds: list, tenors: list):
        for bond in bonds:
            if not isinstance(bond, CashFlowDiscounter):
                msg = 'input bond (%s) is not an instance of %s' % bond.__name__, CashFlowDiscounter.__name__
                logger.error(msg)
                raise TypeError(msg)
        self.bonds = bonds
        self.market_rates = [bond.calc_irr() for bond in bonds]
        self.tenors = tenors
        self.tenors.sort()
        pass

    def fit_zero_curve(self, weights: list):

        pass
