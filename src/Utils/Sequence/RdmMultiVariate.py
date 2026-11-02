import logging

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class RdmMultiVariate(object):
    """
    correlated normal draws: each column of a draw is one sample of n variables, one per row

    rng is anything with standard_normal, np.random (the global state, the default) or a np.random.Generator for
    reproducible draws
    """

    @staticmethod
    def check_correlations_matrix(vcv):
        mtx = np.asarray(vcv, dtype=float)
        if mtx.ndim != 2 or mtx.shape[0] != mtx.shape[1]:
            raise ValueError('variance covariance matrix should be square, received shape %s' % (mtx.shape,))
        if not np.allclose(mtx, mtx.T):
            raise ValueError('variance covariance matrix is not symmetric')

    @staticmethod
    def get_std_correlation(vcv):
        var = np.diag(vcv)
        std = np.sqrt(var)
        correlation = np.asarray(vcv) / np.outer(std, std)
        return std, correlation

    @staticmethod
    def draw_std(correlation, size: int, rng=np.random):
        """
        standard normal variables with these correlations: mean 0 and variance 1 each
        :raises np.linalg.LinAlgError: when the matrix is not positive definite
        """
        RdmMultiVariate.check_correlations_matrix(correlation)
        n = np.shape(correlation)[0]
        cd_l = np.linalg.cholesky(np.asarray(correlation, dtype=float))
        return cd_l @ rng.standard_normal(size=(n, size))

    @staticmethod
    def draw(vcv, size: int, rng=np.random):
        """
        normal variables with mean 0 and this variance covariance matrix: L z, where L L' = vcv and z are independent
        standard normals
        :raises np.linalg.LinAlgError: when the matrix is not positive definite
        """
        RdmMultiVariate.check_correlations_matrix(vcv)
        n = np.shape(vcv)[0]
        cd_l = np.linalg.cholesky(np.asarray(vcv, dtype=float))
        return cd_l @ rng.standard_normal(size=(n, size))
