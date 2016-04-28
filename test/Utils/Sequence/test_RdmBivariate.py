import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
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
    def test_draw_std(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        rho = 0.2
        size = 10 ** 6
        x1, x2 = RdmBivariate.draw_std(rho=rho, size=size)
        cor = np.corrcoef(x1, x2)[(0, 1)]
        logger.debug('rho input: %.6f, simulation output: %.6f' % (rho, cor))
        self.assertAlmostEqual(rho, cor, places=2)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pdf(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        x_1 = np.linspace(-5, 5)
        x_2 = np.linspace(-5, 5)
        pdf = RdmBivariate.pdf(x_1, x_2, rho=-0.45)
        xx_1, xx_2 = np.meshgrid(x_1, x_2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(xx_1, xx_2, pdf)
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
