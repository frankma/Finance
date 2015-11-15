import numpy as np
from src.Utils.BAW import BAW
from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

import unittest


class MyTestCase(unittest.TestCase):
    def test_price(self):
        spots = np.linspace(80, 120, num=5)
        strike = 100
        tau = 0.25
        rate = 0.08
        div = 0.12
        sigma = 0.4

        for spot in spots:
            call_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            call_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.call)
            put_price_ame = BAW.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            put_price_eur = BSM.price(spot, strike, tau, rate, div, sigma, OptionType.put)
            print('%.3f\t%.8f\t%.8f\t%.8f\t%.8f' % (spot, call_price_eur, call_price_ame, put_price_eur, put_price_ame))
