from unittest import TestCase
from src.HestonModel.HestonModel import HestonModel
import matplotlib.pyplot as plt

__author__ = 'frank.ma'


class TestHestonModel(TestCase):
    def test_sim_forward_den(self):
        mu = 0.01
        v_0 = 0.4
        kappa = 0.02
        theta = 0.02
        nu = 0.3
        rho = 0.25

        model = HestonModel(mu, v_0, kappa, theta, nu, rho)
        spot = 150.0
        t = 0.25
        den, bins = model.sim_forward_den(spot, t)

        plt.plot(bins, den)
        plt.show()

        pass
