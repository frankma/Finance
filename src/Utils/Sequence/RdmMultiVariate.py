import logging

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class RdmMultiVariate(object):
    @staticmethod
    def check_correlations_matrix(vcv):
        mtx = np.matrix(vcv)
        mtx_t = mtx.transpose()
        if not (mtx == mtx_t).all():
            raise KeyError('variance covariance matrix is not symmetric')
        pass

    @staticmethod
    def get_std_correlation(vcv):
        var = np.diag(vcv)
        std = np.sqrt(var)
        vec = np.matrix(std)
        correlation = vcv / (vec.transpose() * vec)
        return std, correlation

    @staticmethod
    def draw_std(correlation, size: int):
        RdmMultiVariate.check_correlations_matrix(correlation)
        n = np.shape(correlation)[0]
        randoms = np.matrix(np.random.random(size=(n, size)))
        cd_l = np.linalg.cholesky(correlation)
        return np.array(cd_l * randoms)

    @staticmethod
    def draw(vcv, size: int):
        RdmMultiVariate.check_correlations_matrix(vcv)
        n = np.shape(vcv)[0]
        randoms = np.random.random(size=(n, size))
        std, correlation = RdmMultiVariate.get_std_correlation(vcv)
        cd_l = np.linalg.cholesky(correlation)
        return cd_l * randoms
