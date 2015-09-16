from unittest import TestCase

from src.Utils.NormalModel import NormalModel
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class TestNormalModel(TestCase):

    def test_price(self):
        pass

    def test_imp_vol(self):
        f = 150.0
        k = 150.0
        tau = 0.75
        sig = 0.25 * f
        b = 0.98

        call_price = NormalModel.price(f, k, tau, sig, b, OptionType.call)
        call_imp_vol = NormalModel.imp_vol(f, k, tau, call_price, b, OptionType.call)
        put_price = NormalModel.price(f, k, tau, sig, b, OptionType.put)
        put_imp_vol = NormalModel.imp_vol(f, k, tau, put_price, b, OptionType.put)

        self.assertAlmostEqual(sig, call_imp_vol, places=6, msg='call implied vol search failed')
        self.assertAlmostEqual(sig, put_imp_vol, places=6, msg='put implied vol search failed')
