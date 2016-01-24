import logging
import sys
from unittest import TestCase

import numpy as np
from scipy.stats import norm

from src.Utils.FXQuotesConverter import FXQuotesConverter
from src.Utils.OptionType import OptionType
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestFXQuotesConverter(TestCase):
    def test_read_quotes(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
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
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_vol_to_strike(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        spot = 150.0
        taus = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
        rate_dom = 0.04
        rate_for = 0.02
        strikes = [130.0, 140.0, 150.0, 160.0, 170.0]
        vols = [0.3, 0.25, 0.2, 0.21, 0.26]
        is_atm = [False, False, True, False, False]

        for tau in taus:
            for kdx, strike in enumerate(strikes):
                vol = vols[kdx]
                opt_type = OptionType.call if strike > spot else OptionType.put  # OTM quotes convert much faster
                delta = BSM.delta(spot, strike, tau, rate_dom, rate_for, vol, opt_type=opt_type)
                strike_rep = FXQuotesConverter.vol_to_strike(vol, delta, tau, spot, rate_dom, rate_for, is_atm[kdx],
                                                             is_forward_delta=False)
                self.assertAlmostEqual(strike, strike_rep, places=12,
                                       msg='vol %.4f to strike %.2f at delta %.12f conversion failed at tau %.4f'
                                           % (vol, strike, delta, tau))
                delta *= np.exp(rate_for * tau)  # make it forward delta
                strike_rep = FXQuotesConverter.vol_to_strike(vol, delta, tau, spot, rate_dom, rate_for, is_atm[kdx],
                                                             is_forward_delta=True)
                self.assertAlmostEqual(strike, strike_rep, places=12,
                                       msg='vol %.4f to strike %.2f at delta %.12f conversion failed at tau %.4f'
                                           % (vol, strike, delta, tau))

        for tau in taus:
            for kdx, strike in enumerate(strikes):
                vol = vols[kdx]
                opt_type = OptionType.call if strike > spot else OptionType.put  # OTM quotes convert much faster
                eta = float(opt_type.value)
                d2 = BSM.calc_d2(spot, strike, tau, rate_dom, rate_for, vol)
                delta = strike / spot * eta * np.exp(-rate_for * tau) * norm.cdf(eta * d2)
                strike_rep = FXQuotesConverter.vol_to_strike(vol, delta, tau, spot, rate_dom, rate_for, is_atm[kdx],
                                                             is_forward_delta=False, is_premium_adj=True)
                self.assertAlmostEqual(strike, strike_rep, places=12,
                                       msg='vol %.4f to strike %.2f at delta %.12f conversion failed at tau %.4f'
                                           % (vol, strike, delta, tau))
                delta *= np.exp(rate_for * tau)  # cancel foreign bound to make forward delta
                strike_rep = FXQuotesConverter.vol_to_strike(vol, delta, tau, spot, rate_dom, rate_for, is_atm[kdx],
                                                             is_forward_delta=True, is_premium_adj=True)
                self.assertAlmostEqual(strike, strike_rep, places=12,
                                       msg='vol %.4f to strike %.2f at delta %.12f conversion failed at tau %.4f'
                                           % (vol, strike, delta, tau))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_convert(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        deltas_benchmark = [-0.1, -0.25, 0.5, 0.25, 0.1]
        spot = 1.08865
        tau = 0.019164956
        rate_dom = 0.001318942
        rate_for = -0.003122754
        quotes = {'rr_10': 0.00212848,
                  'rr_25': 0.00184351,
                  'atm_50': 0.101191497,
                  'sm_25': 0.001390179,
                  'sm_10': 0.003914945}

        converter = FXQuotesConverter(spot, tau, rate_dom, rate_for, quotes)
        strikes, vols = converter.convert()
        df_for = np.exp(rate_for * tau)
        deltas = [BSM.delta(spot, strikes[kdx], tau, rate_dom, rate_for, vols[kdx],
                            OptionType.put if strikes[kdx] < spot * df_for else OptionType.call) * df_for
                  for kdx in range(strikes.__len__())]
        for kdx in range(strikes.__len__()):
            self.assertAlmostEqual(deltas_benchmark[kdx], deltas[kdx], places=12)

        tau = 5.002053388
        rate_dom = 0.014129503
        rate_for = -0.004354661
        quotes = {'rr_10': -0.002650901,
                  'rr_25': -0.001451186,
                  'atm_50': 0.105951789,
                  'sm_25': 0.002881223,
                  'sm_10': 0.010264397}

        converter = FXQuotesConverter(spot, tau, rate_dom, rate_for, quotes)
        strikes, vols = converter.convert()
        df_for = np.exp(rate_for * tau)
        deltas = [BSM.delta(spot, strikes[kdx], tau, rate_dom, rate_for, vols[kdx],
                            OptionType.put if strikes[kdx] < spot * df_for else OptionType.call) * df_for
                  for kdx in range(strikes.__len__())]
        for kdx in range(strikes.__len__()):
            self.assertAlmostEqual(deltas_benchmark[kdx], deltas[kdx], places=12)

        tau = 30.00684463
        rate_dom = 0.024621777
        rate_for = 0.011894741
        quotes = {'rr_10': -0.005413679,
                  'rr_25': -0.002916912,
                  'atm_50': 0.12517186,
                  'sm_25': 0.00078497,
                  'sm_10': 0.00606068}

        converter = FXQuotesConverter(spot, tau, rate_dom, rate_for, quotes)
        strikes, vols = converter.convert()
        df_for = np.exp(rate_for * tau)
        deltas = [BSM.delta(spot, strikes[kdx], tau, rate_dom, rate_for, vols[kdx],
                            OptionType.put if strikes[kdx] < spot * df_for else OptionType.call) * df_for
                  for kdx in range(strikes.__len__())]
        for kdx in range(strikes.__len__()):
            self.assertAlmostEqual(deltas_benchmark[kdx], deltas[kdx], places=12)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_convert_premium_adjusted(self):
        pass
