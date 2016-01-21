import logging

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class IUnivariateFunction(object):
    def evaluate(self, x):
        pass


class IMultiVariateFunction(object):
    def evaluate(self, x):
        pass
