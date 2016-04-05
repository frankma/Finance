import logging
import sys
import time as tm
from unittest import TestCase

from src.Utils.Sequence.Prime import PrimeSequence

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestPrimeSequence(TestCase):
    def test_generate_lp(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        n = 10
        ps = PrimeSequence.generate_lp(n)
        tgt = [2, 3, 5, 7]
        self.assertListEqual(tgt, list(ps))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_generate_vec(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        n = 10
        ps = PrimeSequence.generate_vec(n)
        tgt = [2, 3, 5, 7]
        self.assertListEqual(tgt, list(ps))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_reconcile(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        n = 10 ** 4

        tic = tm.time()
        sq_lp = PrimeSequence.generate_lp(n)
        ct_lp = tm.time() - tic
        logger.debug('prime sequence looping method finding primes below %i, elapsed %.2e sec.' % (n, ct_lp))

        tic = tm.time()
        sq_vec = PrimeSequence.generate_vec(n)
        ct_vec = tm.time() - tic
        logger.debug('prime sequence vectorization method finding primes below %i, elapsed %.2e sec.' % (n, ct_vec))

        self.assertLessEqual(ct_vec, ct_lp, 'expected shorter vectorization method calculation time')
        self.assertListEqual(list(sq_lp), list(sq_vec))

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
