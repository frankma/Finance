import logging
import sys

from unittest import TestCase

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestISDAYieldCurve(TestCase):
    def test_calc_df(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
