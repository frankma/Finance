import logging
import sys
from unittest import TestCase

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


class TestBSM(TestCase):
    def test_price(self):
        pass

    def test_imp_vol(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        s = 110
        k = 115
        tau = 0.25
        r = 0.02
        q = 0.01
        sig = 0.66

        call_price = BSM.price(s, k, tau, r, q, sig, OptionType.call)
        put_price = BSM.price(s, k, tau, r, q, sig, OptionType.put)

        call_imp_vol = BSM.imp_vol(s, k, tau, r, q, call_price, OptionType.call)
        put_imp_vol = BSM.imp_vol(s, k, tau, r, q, put_price, OptionType.put)

        self.assertAlmostEqual(sig, call_imp_vol, places=6, msg='call implied vol search failed')
        self.assertAlmostEqual(sig, put_imp_vol, places=6, msg='put implied vol search failed')
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
