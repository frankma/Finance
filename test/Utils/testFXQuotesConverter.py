from unittest import TestCase

from src.Utils.FXQuotesConverter import FXQuotesConverter

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

        self.assertAlmostEqual(sig_10, res[0], places=12, msg='delta 0.9 par failed')
        self.assertAlmostEqual(sig_25, res[1], places=12, msg='delta 0.9 par failed')
        self.assertAlmostEqual(sig_50, res[2], places=12, msg='delta 0.9 par failed')
        self.assertAlmostEqual(sig_75, res[3], places=12, msg='delta 0.9 par failed')
        self.assertAlmostEqual(sig_90, res[4], places=12, msg='delta 0.9 par failed')
        pass
