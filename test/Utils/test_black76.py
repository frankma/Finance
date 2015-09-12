from unittest import TestCase
from src.Utils.Black76 import Black76
from src.Utils.OptionType import OptionType
from math import log

__author__ = 'frank.ma'


class TestBlack76(TestCase):

    def test_price(self):
        forward = 110.
        strike = 123.
        ttm = 0.75
        sigma = 0.45
        bond = 0.92

        call_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.call)
        put_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.put)

        lhs = call_price - put_price
        rhs = bond * (forward - strike)

        self.assertAlmostEqual(lhs, rhs, places=12, msg='call put parity check failed')

    def test_imp_vol(self):
        forward = 110.
        strike = 123.
        ttm = 0.75
        sigma = 0.45
        bond = 0.92

        call_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.call)
        put_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.put)

        call_imp_vol = Black76.imp_vol(forward, strike, ttm, call_price, bond, OptionType.call)
        put_imp_vol = Black76.imp_vol(forward, strike, ttm, put_price, bond, OptionType.put)

        self.assertAlmostEqual(sigma, call_imp_vol, places=6, msg='call implied vol search failed')
        self.assertAlmostEqual(sigma, put_imp_vol, places=6, msg='put implied vol search failed')

    def test_pde(self):
        forward = 110.
        strike = 123.
        ttm = 0.75
        sigma = 0.45
        bond = 0.92

        r = -log(bond) / ttm

        call_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.call)
        put_price = Black76.price(forward, strike, ttm, sigma, bond, OptionType.put)

        call_gamma = Black76.gamma(forward, strike, ttm, sigma, bond)
        put_gamma = call_gamma

        call_theta = Black76.theta(forward, strike, ttm, sigma, bond, OptionType.call)
        put_theta = Black76.theta(forward, strike, ttm, sigma, bond, OptionType.put)

        call_lhs = call_theta + 0.5 * call_gamma * sigma**2 * forward**2
        put_lhs = put_theta + 0.5 * put_gamma * sigma**2 * forward**2

        call_rhs = r * call_price
        put_rhs = r * put_price

        self.assertAlmostEqual(call_lhs, call_rhs, places=6, msg='call PDE check failed')
        self.assertAlmostEqual(put_lhs, put_rhs, places=6, msg='put PDE check failed')
