from unittest import TestCase
from math import log
import numpy as np
from src.Utils.Black76 import Black76, Black76Vec
from src.Utils.OptionType import OptionType

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

        call_lhs = call_theta + 0.5 * call_gamma * sigma ** 2 * forward ** 2
        put_lhs = put_theta + 0.5 * put_gamma * sigma ** 2 * forward ** 2

        call_rhs = r * call_price
        put_rhs = r * put_price

        self.assertAlmostEqual(call_lhs, call_rhs, places=6, msg='call PDE check failed')
        self.assertAlmostEqual(put_lhs, put_rhs, places=6, msg='put PDE check failed')


class TestBlack76Vec(TestCase):
    def test_price_vec(self):
        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=21)
        ttm = 0.75
        sigmas = np.full(21, 0.45)
        bond = 0.92

        call_prices = Black76Vec.price(forward, strikes, ttm, sigmas, bond, OptionType.call)
        put_prices = Black76Vec.price(forward, strikes, ttm, sigmas, bond, OptionType.put)

        for idx, strike in enumerate(strikes):
            call_price = Black76.price(forward, strike, ttm, sigmas[idx], bond, OptionType.call)
            put_price = Black76.price(forward, strike, ttm, sigmas[idx], bond, OptionType.put)
            self.assertAlmostEqual(call_price, call_prices[idx], places=12, msg='call vectorization differs')
            self.assertAlmostEqual(put_price, put_prices[idx], places=12, msg='put vectorization differs')
        pass

    def test_imp_vol_vec(self):
        forward = 150.0
        strikes = np.linspace(100.0, 200.0, num=21)
        ttm = 0.75
        sigmas = 0.0001 * ((strikes - forward) ** 2) + 0.25  # use polynomial to define vol smile
        bond = 0.92

        call_prices = Black76Vec.price(forward, strikes, ttm, sigmas, bond, OptionType.call)
        put_prices = Black76Vec.price(forward, strikes, ttm, sigmas, bond, OptionType.put)

        call_vols = Black76Vec.imp_vol(forward, strikes, ttm, call_prices, bond, OptionType.call)
        put_vols = Black76Vec.imp_vol(forward, strikes, ttm, put_prices, bond, OptionType.put)

        for idx, sigma in enumerate(sigmas):
            self.assertAlmostEqual(sigma, call_vols[idx], places=6, msg='call vol regression failed')
            self.assertAlmostEqual(sigma, put_vols[idx], places=6, msg='call vol regression failed')
        pass

    def test_backward_pde(self):
        pass

    def test_forward_pde(self):
        pass
