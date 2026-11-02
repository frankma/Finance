import logging
import sys
from unittest import TestCase

import numpy as np

from src.Utils.Sequence.RdmMultiVariate import RdmMultiVariate

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestRdmMultiVariate(TestCase):
    def test_check_correlations_matrix(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        for bad in ([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3]], [[1.0, 0.2], [0.3, 1.0]]):
            with self.assertRaises(ValueError):
                RdmMultiVariate.check_correlations_matrix(bad)
        # symmetric up to rounding, as a matrix built in code often is
        RdmMultiVariate.check_correlations_matrix([[1.0, 0.1 + 0.2], [0.3, 1.0]])
        with self.assertRaises(np.linalg.LinAlgError):
            RdmMultiVariate.draw([[1.0, 2.0], [2.0, 1.0]], size=10)  # not positive definite
        logger.info('%s passes' % sys._getframe().f_code.co_name)

    def test_draw(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        vcv = np.array([[2.0, 1.3], [1.3, 3.0]])
        n = 2 * 10 ** 5
        rnd = RdmMultiVariate.draw(vcv, size=n, rng=np.random.default_rng(1))
        self.assertEqual(rnd.shape, (2, n))
        std = np.sqrt(np.diag(vcv))
        # each mean within 4 standard errors of 0, the covariance within 2% of the input
        np.testing.assert_array_less(np.abs(rnd.mean(axis=1)), 4.0 * std / np.sqrt(n))
        np.testing.assert_allclose(np.cov(rnd), vcv, rtol=0.02)
        # normal, not uniform: about 68.3% of each variable within one standard deviation of the mean
        within = np.mean(np.abs(rnd) < std[:, None], axis=1)
        np.testing.assert_allclose(within, 0.6827, atol=0.005)
        logger.info('%s passes' % sys._getframe().f_code.co_name)

    def test_draw_std(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        correlation = [[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]]
        rnd = RdmMultiVariate.draw_std(correlation, size=2 * 10 ** 5, rng=np.random.default_rng(2))
        np.testing.assert_allclose(rnd.mean(axis=1), 0.0, atol=0.01)
        np.testing.assert_allclose(rnd.std(axis=1), 1.0, atol=0.01)
        np.testing.assert_allclose(np.corrcoef(rnd), correlation, atol=0.01)
        logger.info('%s passes' % sys._getframe().f_code.co_name)

    def test_draws_repeat_with_a_seed(self):
        vcv = [[2.0, 1.3], [1.3, 3.0]]
        a = RdmMultiVariate.draw(vcv, size=5, rng=np.random.default_rng(3))
        b = RdmMultiVariate.draw(vcv, size=5, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_get_std_correlation(self):
        std, correlation = RdmMultiVariate.get_std_correlation([[4.0, 1.2], [1.2, 9.0]])
        np.testing.assert_allclose(std, [2.0, 3.0])
        np.testing.assert_allclose(correlation, [[1.0, 0.2], [0.2, 1.0]])
