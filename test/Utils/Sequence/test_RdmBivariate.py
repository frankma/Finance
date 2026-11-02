import logging
import sys
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

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
        x1, x2 = RdmBivariate.draw_std(rho=rho, size=size, rng=np.random.default_rng(0))
        cor = np.corrcoef(x1, x2)[(0, 1)]
        logger.debug('rho input: %.6f, simulation output: %.6f' % (rho, cor))
        self.assertAlmostEqual(rho, cor, places=2)
        # standard normals: mean 0, standard deviation 1, about 68.3% within one of 0
        for x in (x1, x2):
            self.assertAlmostEqual(x.mean(), 0.0, places=2)
            self.assertAlmostEqual(x.std(), 1.0, places=2)
            self.assertAlmostEqual(np.mean(np.abs(x) < 1.0), 0.6827, places=2)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_pdf(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        x_1 = np.linspace(-5, 5)
        x_2 = np.linspace(-5, 5)
        pdf = RdmBivariate.pdf(x_1, x_2, rho=-0.45)
        xx_1, xx_2 = np.meshgrid(x_1, x_2)
        fig = plt.figure()
        Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(xx_1, xx_2, pdf)
        plt.show()
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass


    def test_pdf_values(self):
        # against scipy's bivariate normal, with means and scales away from the standard case
        rho, mu_1, mu_2, sig_1, sig_2 = -0.45, 0.3, -1.0, 1.5, 0.7
        x_1 = np.linspace(-6.0, 6.0, 241)
        x_2 = np.linspace(-5.0, 3.0, 161)
        pdf = RdmBivariate.pdf(x_1, x_2, rho=rho, mu_1=mu_1, mu_2=mu_2, sig_1=sig_1, sig_2=sig_2)
        xx_1, xx_2 = np.meshgrid(x_1, x_2)
        cov = [[sig_1 ** 2, rho * sig_1 * sig_2], [rho * sig_1 * sig_2, sig_2 ** 2]]
        expected = multivariate_normal(mean=[mu_1, mu_2], cov=cov).pdf(np.dstack([xx_1, xx_2]))
        np.testing.assert_allclose(pdf, expected, rtol=1e-10, atol=1e-14)
        # and it integrates to 1 over a grid that covers the mass
        self.assertAlmostEqual(np.trapezoid(np.trapezoid(pdf, x_1, axis=1), x_2), 1.0, places=3)
