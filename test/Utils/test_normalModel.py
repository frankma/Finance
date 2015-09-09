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

        v_call = NormalModel.price(f, k, tau, sig, b, OptionType.call)
        vol_call = NormalModel.imp_vol(f, k, tau, v_call, b, OptionType.call)
        v_put = NormalModel.price(f, k, tau, sig, b, OptionType.put)
        vol_put = NormalModel.imp_vol(f, k, tau, v_put, b, OptionType.put)

        assert abs(vol_call / sig - 1.0) < 1e-4, "call imp vol search failed"
        assert abs(vol_put / sig - 1.0) < 1e-4, "put imp vol search failed"
