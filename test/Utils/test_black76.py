from unittest import TestCase
from src.Utils.Black76 import Black76, OptionType
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

        assert abs(lhs / rhs - 1.) < 1e-10, 'call put parity violated.'
        pass

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

        assert abs(call_imp_vol / sigma - 1) < 1e-6, 'call implied vol search failed.'
        assert abs(put_imp_vol / sigma - 1) < 1e-6, 'put implied vol search failed.'
        pass

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

        assert abs(call_lhs / call_rhs - 1.) < 1e-6, 'call pde check failed.'
        assert abs(put_lhs / put_rhs - 1.) < 1e-6, 'put pde check failed.'