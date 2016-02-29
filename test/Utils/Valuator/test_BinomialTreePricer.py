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
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestBinomialTreePricer(TestCase):
    def test_calc_u_d_p(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        dts = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        rs = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        qs = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        sigs = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
        for dt in dts:
            for r in rs:
                for q in qs:
                    for sig in sigs:
                        u, d, p = BinomialTreePricer.calc_u_d_p(dt, r, q, sig)
                        label = 'dt: %.8f, r: %.8f, q: %.8f, sig: %.8f' % (dt, r, q, sig)
                        self.assertAlmostEqual(1.0, u * d, places=12,
                                               msg='up and down sanity check failed with params %s' % label)
                        self.assertAlmostEqual(1.0, np.exp(-dt * (r - q)) * (u * p + d * (1.0 - p)), places=12,
                                               msg='probability sanity check failed with params %s' % label)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

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

    def test_time_march(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        s, k, tau, r, q, sig, opt_type, n_steps = 100.0, 102.0, 2.0, 0.01, 0.1, 0.6, OptionType.call, 1000
        btp = BinomialTreePricer(s, k, tau, r, q, sig, opt_type, n_steps)
        opt_ame_num = btp.price_ame_opt()
        opt_ame_ana = BAW.price(s, k, tau, r, q, sig, opt_type)
        print('Option Type\tAnalytical Price\tNumerical Price\tDifference')
        print('American:\t%.6f\t%.6f\t%.4e' % (opt_ame_ana, opt_ame_num, opt_ame_num / opt_ame_ana - 1.0))

        opt_eur_num = btp.price_eur_opt()
        opt_eur_ana = BSM.price(s, k, tau, r, q, sig, opt_type)
        print('European:\t%.6f\t%.6f\t%.4e' % (opt_eur_ana, opt_eur_num, opt_eur_num / opt_eur_ana - 1.0))

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
