import logging
import sys
from unittest import TestCase

from src.Utils.Sequence.Fibonacci import Fibonacci

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestFibonacci(TestCase):
    def test_generate_list(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        num = 10
        tgt = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        seq = Fibonacci.generate_list(num)
        self.assertListEqual(tgt, seq)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_generate_nth(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        num = 100
        seq = Fibonacci.generate_list(num)
        for idx in range(2, num):
            value = Fibonacci.generate_nth(idx)
            self.assertEquals(seq[idx - 1], value, 'value disagree on methods list %i and scalar %i' % (seq[idx], value))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
