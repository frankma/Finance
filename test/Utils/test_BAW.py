from src.Utils.BAW import BAW
from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

import unittest


class MyTestCase(unittest.TestCase):
    def test_price(self):
        s = 150.0
        k = 155.0
        tau = 0.5
        r = 0.02
        q = 0.1
        sig = 0.45
        opt_type = OptionType.call

        price_ame = BAW.price(s, k, tau, r, q, sig, opt_type)
        price_eur = BSM.price(s, k, tau, r, q, sig, opt_type)

        print(price_eur, price_ame)
