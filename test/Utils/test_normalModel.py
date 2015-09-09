from unittest import TestCase
from src.Utils.NormalModel import NormalModel
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class TestNormalModel(TestCase):

    def test_price(self):
        pass

    def test_normal_vol(self):
        f = 150.0
        k = 150.0
        tau = 0.75
        sig = 0.05
        b = 0.98
        opt_type = OptionType.call
        v = NormalModel.price(f, k, tau, sig, b, opt_type)
        vol = NormalModel.normal_vol(f, k, tau, v, b, opt_type)
        print(sig, vol)
