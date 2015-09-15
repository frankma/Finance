import numpy as np
import matplotlib.pylab as plt

from unittest import TestCase
from src.Utils.Solver.RealNumberTransformer import RealNumberTransformer


__author__ = 'frank.ma'


class TestRealNumberTransformer(TestCase):

    def test_uc_to_c(self):

        x = np.array(range(-200, 200)) * 0.1
        lower_bound = 0.0
        upper_bound = 4.0

        rnt_none = RealNumberTransformer(None, None)
        self.assertEquals(1.0, rnt_none.uc_to_c(1.0))

        rnt_abs = RealNumberTransformer(lower_bound, upper_bound, 'abs')
        rnt_tan = RealNumberTransformer(lower_bound, upper_bound, 'tan')
        rnt_norm = RealNumberTransformer(lower_bound, upper_bound, 'norm')
        rnt_lb = RealNumberTransformer(lower_bound, None)
        rnt_ub = RealNumberTransformer(None, upper_bound)

        y_abs = x.copy()
        x_abs = x.copy()
        y_tan = x.copy()
        x_tan = x.copy()
        y_norm = x.copy()
        x_norm = x.copy()
        y_lb = x.copy()
        x_lb = x.copy()
        y_ub = x.copy()
        x_ub = x.copy()

        for idx, v in enumerate(x):
            y_abs[idx] = rnt_abs.uc_to_c(v)
            x_abs[idx] = rnt_abs.c_to_uc(y_abs[idx])
            y_tan[idx] = rnt_tan.uc_to_c(v)
            x_tan[idx] = rnt_tan.c_to_uc(y_tan[idx])
            y_norm[idx] = rnt_norm.uc_to_c(v)
            x_norm[idx] = rnt_norm.c_to_uc(x_norm[idx])
            y_lb[idx] = rnt_lb.uc_to_c(v)
            x_lb[idx] = rnt_lb.c_to_uc(y_lb[idx])
            y_ub[idx] = rnt_ub.uc_to_c(v)
            x_ub[idx] = rnt_ub.c_to_uc(y_ub[idx])
            self.assertAlmostEqual(v, x_abs[idx], places=12, msg='abs method check failed.')
            self.assertAlmostEqual(v, x_tan[idx], places=12, msg='tan method check failed.')
            # self.assertAlmostEqual(v, x_norm[idx], places=12, msg='norma method check failed.')
            self.assertAlmostEqual(v, x_lb[idx], places=6, msg='lower bound log method check failed.')
            self.assertAlmostEqual(v, x_ub[idx], places=6, msg='upper bound log method check failed.')

    def test_c_to_uc(self):

        y = np.array(range(0, 4)) * 0.01
        lower_bound = -1e-4
        upper_bound = 4.0 + 1e-4

        rnt_none = RealNumberTransformer(None, None)
        self.assertEquals(1.0, rnt_none.c_to_uc(1.0))

        rnt_abs = RealNumberTransformer(lower_bound, upper_bound, 'abs')
        rnt_tan = RealNumberTransformer(lower_bound, upper_bound, 'tan')
        rnt_norm = RealNumberTransformer(lower_bound, upper_bound, 'norm')
        rnt_lb = RealNumberTransformer(lower_bound, None)
        rnt_ub = RealNumberTransformer(None, upper_bound)

        y_abs = y.copy()
        x_abs = y.copy()
        y_tan = y.copy()
        x_tan = y.copy()
        y_norm = y.copy()
        x_norm = y.copy()
        y_lb = y.copy()
        x_lb = y.copy()
        y_ub = y.copy()
        x_ub = y.copy()

        for idx, v in enumerate(y):
            x_abs[idx] = rnt_abs.c_to_uc(v)
            y_abs[idx] = rnt_abs.uc_to_c(x_abs[idx])
            x_tan[idx] = rnt_tan.c_to_uc(v)
            y_tan[idx] = rnt_tan.uc_to_c(x_tan[idx])
            x_norm[idx] = rnt_norm.c_to_uc(v)
            y_norm[idx] = rnt_norm.uc_to_c(x_norm[idx])
            x_lb[idx] = rnt_lb.c_to_uc(v)
            y_lb[idx] = rnt_lb.uc_to_c(x_lb[idx])
            x_ub[idx] = rnt_ub.c_to_uc(v)
            y_ub[idx] = rnt_ub.uc_to_c(x_ub[idx])
            self.assertAlmostEqual(v, y_abs[idx], places=12, msg='abs method check failed.')
            self.assertAlmostEqual(v, y_tan[idx], places=12, msg='tan method check failed.')
            # self.assertAlmostEqual(v, y_norm[idx], places=12, msg='norm method check failed.')
            self.assertAlmostEqual(v, y_lb[idx], places=6, msg='norm method check failed.')
            self.assertAlmostEqual(v, y_ub[idx], places=6, msg='norm method check failed.')

    def test_graphical(self):

        x = np.arange(-20.0, 20.0, 0.1)
        lower_bound = -1.0
        upper_bound = 4.0

        rnt_abs = RealNumberTransformer(lower_bound, upper_bound, 'abs')
        rnt_log = RealNumberTransformer(lower_bound, upper_bound, 'tan')
        rnt_norm = RealNumberTransformer(lower_bound, upper_bound, 'norm')

        y_abs = x.copy()
        y_log = x.copy()
        y_norm = x.copy()

        for idx, v in enumerate(x):
            y_abs[idx] = rnt_abs.uc_to_c(v)
            y_log[idx] = rnt_log.uc_to_c(v)
            y_norm[idx] = rnt_norm.uc_to_c(v)

        plt.plot(x, y_abs, 'r', x, y_log, 'b', x, y_norm, 'k')
        plt.legend(('absolute', 'arc-tangent', 'normal-inverse'))
        plt.show()
        pass
