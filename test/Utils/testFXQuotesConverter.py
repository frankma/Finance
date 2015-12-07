from unittest import TestCase

from src.Utils.BSM import BSM
from src.Utils.FXQuotesConverter import FXQuotesConverter
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class TestFXQuotesConverter(TestCase):
    def test_read_quotes(self):
        sig_10, sig_25, sig_50, sig_75, sig_90 = 0.40, 0.35, 0.30, 0.33, 0.34

        quotes = {'rr_10': sig_90 - sig_10,
                  'rr_25': sig_75 - sig_25,
                  'atm_50': sig_50,
                  'sm_25': 0.5 * (sig_25 - 2.0 * sig_50 + sig_75),
                  'sm_10': 0.5 * (sig_10 - 2.0 * sig_50 + sig_90)}

        res = FXQuotesConverter.read_quotes(quotes)

        self.assertAlmostEqual(sig_10, res[0], places=12, msg='delta 0.10 conversion failed')
        self.assertAlmostEqual(sig_25, res[1], places=12, msg='delta 0.25 conversion failed')
        self.assertAlmostEqual(sig_50, res[2], places=12, msg='delta 0.50 conversion failed')
        self.assertAlmostEqual(sig_75, res[3], places=12, msg='delta 0.75 conversion failed')
        self.assertAlmostEqual(sig_90, res[4], places=12, msg='delta 0.90 conversion failed')

        sig_05 = 0.50
        sig_95 = 0.40
        quotes.update({'rr_05': sig_95 - sig_05,
                       'sm_05': 0.5 * (sig_05 - 2.0 * sig_50 + sig_95)})
        res = FXQuotesConverter.read_quotes(quotes, seven_quotes=True)

        self.assertAlmostEqual(sig_05, res[0], places=12, msg='delta 0.05 conversion failed')
        self.assertAlmostEqual(sig_10, res[1], places=12, msg='delta 0.10 conversion failed')
        self.assertAlmostEqual(sig_25, res[2], places=12, msg='delta 0.25 conversion failed')
        self.assertAlmostEqual(sig_50, res[3], places=12, msg='delta 0.50 conversion failed')
        self.assertAlmostEqual(sig_75, res[4], places=12, msg='delta 0.75 conversion failed')
        self.assertAlmostEqual(sig_90, res[5], places=12, msg='delta 0.90 conversion failed')
        self.assertAlmostEqual(sig_95, res[6], places=12, msg='delta 0.95 conversion failed')
        pass

    def test_vol_to_strike(self):
        spot = 150.0
        taus = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
        rate_dom = 0.04
        rate_for = 0.02
        strikes = [50.0, 100.0, 150.0, 200.0, 250.0]
        vols = [0.3, 0.25, 0.2, 0.21, 0.26]

        for tau in taus:
            for kdx, strike in enumerate(strikes):
                vol = vols[kdx]
                opt_type = OptionType.call if strike > spot else OptionType.put  # OTM quotes convert much faster
                delta = BSM.delta(spot, strike, tau, rate_dom, rate_for, vol, opt_type=opt_type)
                strike_rep = FXQuotesConverter.vol_to_strike(vol, delta, tau, spot, rate_dom, rate_for)
                self.assertAlmostEqual(strike, strike_rep, places=12,
                                       msg='vol %.4f to strike %.2f at delta %.12f conversion failed at tau %.4f'
                                           % (vol, strike, delta, tau))
        pass

    def test_convert(self):
        spot = 1.10265
        tau = 0.498288843
        rate_dom = 0.00262834
        rate_for = -0.005218988
        quotes = {'rr_10': -0.030858411,
                  'rr_25': -0.018367746,
                  'atm_50': 0.103690639,
                  'sm_25': 0.002870256,
                  'sm_10': 0.008569103}

        converter = FXQuotesConverter(spot, tau, rate_dom, rate_for, quotes)
        strikes, vols = converter.convert()

        print(strikes)
        print(vols)
        pass
