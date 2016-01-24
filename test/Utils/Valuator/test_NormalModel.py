import logging
import sys
from unittest import TestCase

from src.Utils.OptionType import OptionType
from src.Utils.Valuator.NormalModel import NormalModel

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestNormalModel(TestCase):
    def test_price(self):
        pass

    def test_imp_vol(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        f = 150.0
        k = 150.0
        tau = 0.75
        sig = 0.25 * f
        b = 0.98

        call_price = NormalModel.price(f, k, tau, sig, b, OptionType.call)
        call_imp_vol = NormalModel.imp_vol(f, k, tau, call_price, b, OptionType.call)
        put_price = NormalModel.price(f, k, tau, sig, b, OptionType.put)
        put_imp_vol = NormalModel.imp_vol(f, k, tau, put_price, b, OptionType.put)

        self.assertAlmostEqual(sig, call_imp_vol, places=6, msg='call implied vol search failed')
        self.assertAlmostEqual(sig, put_imp_vol, places=6, msg='put implied vol search failed')
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
