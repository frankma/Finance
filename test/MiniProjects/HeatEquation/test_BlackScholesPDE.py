from unittest import TestCase

from src.MiniProjects.HeatEquation.BlackScholesBackwardPDE import BlackScholesBackwardPDE
from src.MiniProjects.HeatEquation.BlackScholesForwardPDE import BlackScholesForwardPDE
from src.Utils.Types.OptionType import OptionType
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'


class TestBlackScholesPDE(TestCase):
    def test_both_match_the_closed_form(self):
        # a call by Crank-Nicolson, backward in the spot and forward in the strike, against Black-Scholes
        t, s, r, sig = 0.5, 150.0, 0.05, 0.45
        for k in (120.0, 150.0, 180.0):
            exact = BSM.price(s, k, t, r, 0.0, sig, OptionType.call)
            backward = BlackScholesBackwardPDE(t, s, k, r, sig, n_x=512, n_t=64).solve()
            forward = BlackScholesForwardPDE(t, s, k, r, sig, n_x=512, n_t=64).solve()
            self.assertAlmostEqual(backward, exact, delta=0.005, msg='backward, strike %.0f' % k)
            self.assertAlmostEqual(forward, exact, delta=0.005, msg='forward, strike %.0f' % k)
            self.assertAlmostEqual(backward, forward, delta=1e-8, msg='the two agree for a flat volatility')

    def test_finer_grids_converge(self):
        t, s, k, r, sig = 0.5, 150.0, 150.0, 0.05, 0.45
        exact = BSM.price(s, k, t, r, 0.0, sig, OptionType.call)
        errors = [abs(BlackScholesBackwardPDE(t, s, k, r, sig, n_x=n_x, n_t=n_t).solve() - exact)
                  for n_x, n_t in ((64, 16), (256, 32), (512, 64))]
        self.assertEqual(errors, sorted(errors, reverse=True))
