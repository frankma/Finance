import logging
import sys

from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.Utils.Sequence.RdmBivariate import RdmBivariate

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestRdmBivariate(TestCase):
    def test_draw(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pdf(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        x_1 = np.linspace(-10, 10)
        x_2 = np.linspace(-10, 10)
        pdf = RdmBivariate.pdf(x_1, x_2, rho=0.1)
        xx_1, xx_2 = np.meshgrid(x_1, x_2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(xx_1, xx_2, pdf)
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
