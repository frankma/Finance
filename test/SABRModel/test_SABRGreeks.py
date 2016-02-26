from unittest import TestCase

import numpy as np

from src.SABRModel.SABRGreeks import SABRGreeks
from src.SABRModel.SABRModel import SABRModelLognormalApprox
from src.Utils.Interpolator.LinearInterpolator1D import LinearInterpolator1D
import matplotlib.pyplot as plt

__author__ = 'frank.ma'


class TestSABRGreeks(TestCase):
    def test_pde(self):
        forward = 150.0
        strikes = np.linspace(50.0, 250.0, num=21)
        tau = 0.75
        b = 0.93
        alpha, beta, nu, rho = 0.2, 1.0, 0.4, -0.33
        model = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        density, bins = model.calc_fwd_den_sp(forward, rel_bounds=(0.1, 2.0))
        interpolator = LinearInterpolator1D(bins, density)
        gamma_k = SABRGreeks.gamma_k(forward, strikes, tau, b, model)
        for kdx, strike in enumerate(strikes):
            den = interpolator.calc(strike)
            print(strike, den, gamma_k[kdx])
        plt.plot(strikes, gamma_k, 'x')
        plt.plot(bins, density, '-')
        plt.show()
