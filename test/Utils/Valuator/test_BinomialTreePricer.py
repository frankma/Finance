import logging
import sys
from unittest import TestCase

import numpy as np

from src.Utils.OptionType import OptionType
from src.Utils.Valuator.BAW import BAW
from src.Utils.Valuator.BSM import BSM
from src.Utils.Valuator.BinomialTreePricer import BinomialTreePricer

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestBinomialTreePricer(TestCase):
    def test_create_state(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        s_s = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        n_steps_s = [1, 49, 99, 499]
        for s in s_s:
            for n_steps in n_steps_s:
                state = BinomialTreePricer.create_final_state(s, n_steps, 1.01)
                origins = np.sqrt(state * state[::-1])
                for idx, origin in enumerate(origins):
                    self.assertAlmostEqual(s, origin, places=12, msg='sanity check failed at %i' % idx)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_price_european_option(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        tau, r, q, n_steps = 2.0, 0.02, 0.04, 99
        logger.debug('testing instrument name\tanalytical\tnumerical\tdiff')
        for s in [140.0, 150.0, 160.0]:
            for k in [140.0, 150.0, 160.0]:
                for sig in [0.4, 0.5, 0.6]:
                    for opt_type in [OptionType.put, OptionType.call]:
                        num = BinomialTreePricer.price_european_option(s, k, tau, r, q, sig, opt_type, n_steps)
                        ana = BSM.price(s, k, tau, r, q, sig, opt_type)
                        self.assertAlmostEqual(1.0, num / ana, places=2)
                        logger.debug('s_%.2f_k_%.2f_vol_%.2f_%s\t%.6f\t%.6f\t%.2e'
                                     % (s, k, sig, opt_type.name[0], ana, num, num / ana - 1.0))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_price_american_option(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        tau, r, q, n_steps = 2.0, 0.02, 0.04, 99
        logger.debug('testing instrument name\tanalytical\tnumerical\tdiff')
        for s in [140.0, 150.0, 160.0]:
            for k in [140.0, 150.0, 160.0]:
                for sig in [0.4, 0.5, 0.6]:
                    for opt_type in [OptionType.put, OptionType.call]:
                        num = BinomialTreePricer.price_american_option(s, k, tau, r, q, sig, opt_type, n_steps)
                        ana = BAW.price(s, k, tau, r, q, sig, opt_type)
                        self.assertAlmostEqual(1.0, num / ana, places=1)
                        logger.debug('s_%.2f_k_%.2f_vol_%.2f_%s\t%.6f\t%.6f\t%.2e'
                                     % (s, k, sig, opt_type.name[0], ana, num, num / ana - 1.0))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
