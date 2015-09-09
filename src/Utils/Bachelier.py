from src.Utils.NormalModel import NormalModel
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class Bachelier(object):

    @staticmethod
    def price(f: float, k: float, tau: float, sig: float, b: float, opt_type: OptionType):
        return NormalModel.price(f, k, tau, sig / f, b, opt_type)
