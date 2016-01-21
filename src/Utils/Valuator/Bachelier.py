import logging

from src.Utils.OptionType import OptionType
from src.Utils.Valuator.NormalModel import NormalModel

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Bachelier(object):
    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        return NormalModel.price(f, k, tau, sig / f, b, opt_type)

    @staticmethod
    def delta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        return NormalModel.delta(f, k, tau, sig / f, b, opt_type)

    @staticmethod
    def vega(f: float, k: float, tau: float, sig: float, b: float):
        return NormalModel.vega(f, k, tau, sig / f, b)

    @staticmethod
    def theta(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        return NormalModel.theta(f, k, tau, sig / f, b, opt_type)

    @staticmethod
    def gamma(f: float, k: float, tau: float, sig: float, b: float):
        return NormalModel.gamma(f, k, tau, sig / f, b)
