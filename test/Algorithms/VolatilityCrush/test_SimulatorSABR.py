import logging
import sys
from unittest import TestCase

import numpy as np
from datetime import datetime, timedelta

from src.Algorithms.VolatilityCrush.SimulatorSABR import SimulatorSABR
from src.SABRModel.SABRModel import SABRModelLognormalApprox

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestSimulatorSABR(TestCase):
    def test_model_shock(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_display_vol_shocks(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        dt_pre = datetime(2016, 2, 16, 16, 0, 0)
        dt_event = datetime(2016, 2, 17, 16, 0, 0)
        dt_post = datetime(2016, 2, 18, 16, 0, 0)

        exp_s = datetime(2016, 2, 19)
        exp_l = datetime(2016, 3, 18)

        tau_s = (exp_s - dt_pre) / timedelta(365.25)
        tau_l = (exp_l - dt_pre) / timedelta(365.25)

        print(tau_s, tau_l)

        r = 0.05
        q = 0.03

        sig_s = 0.55
        sig_l = 0.45

        strike = 155.0
        spot_pre = 150.0
        spot_post = 170.0

        tau, alpha, beta, nu, rho = 15.0 / 365.25, 0.7, 1.0, 2.2, -0.4
        forward, zb = 100.0, np.exp(-0.01 * tau)
        model_pre = SABRModelLognormalApprox(tau, alpha, beta, nu, rho)
        simulator = SimulatorSABR(model_pre, forward, zb)
        alpha_shift, nu_shift, rho_shift = 0.88, 1.5, 1.0
        simulator.display_vol_shocks(alpha_shift=alpha_shift, nu_shift=nu_shift, rho_shift=rho_shift)

        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_strategy_single_period(self):
        pass

    def test_simulate(self):
        pass
