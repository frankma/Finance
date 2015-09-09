from src.Utils.NormalModel import NormalModel
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


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
