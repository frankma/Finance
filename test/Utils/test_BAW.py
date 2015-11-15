import numpy as np
from src.Utils.BAW import BAW
from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

import unittest


class MyTestCase(unittest.TestCase):
    def test_price(self):
        spots = np.linspace(0.97, 0.978, num=9)
        strike = 0.9
        tau = 0.25
        rate = 0.02
        div = 0.035
        sigma = 0.1
        opt_type = OptionType.put

        for spot in spots:
            price_ame = BAW.price(spot, strike, tau, rate, div, sigma, opt_type)
            price_eur = BSM.price(spot, strike, tau, rate, div, sigma, opt_type)
            print('%.3f\t%.8f\t%.8f' % (spot, price_eur, price_ame))
