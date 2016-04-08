import logging
import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SimpleLinearRegression(object):
    @staticmethod
    def regress(independent: np.array, dependent: np.array):
        n = independent.__len__()
        if n != dependent.__len__():
            raise ValueError('expect same length of inputs')
        xy = independent * dependent
        xx = dependent * dependent
        x_mean = np.average(independent)
        y_mean = np.average(dependent)
        slope = (np.average(xy) - x_mean * y_mean) / (np.average(xx) - x_mean * x_mean)
        interception = y_mean - slope * x_mean
        return interception, slope

    def __init__(self, independent: list or np.array, dependent: list or np.array):
        alpha, beta = self.regress(np.array(independent), np.array(dependent))
        self.alpha = alpha
        self.beta = beta
        pass

    def predict(self, x: float or np.array):
        return self.alpha + self.beta * x
