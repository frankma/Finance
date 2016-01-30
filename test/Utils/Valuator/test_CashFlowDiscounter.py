import logging
import sys
from unittest import TestCase

import numpy as np

from src.Utils.Valuator.CashFlowDiscounter import CashFlowDiscounter

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestCashFlowDiscounter(TestCase):
    def test_calc_present_values(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # case of zero discount rate and zero npv
        taus = np.array([0.0, 1.0])
        cash_flows = np.array([-1.0, 1.0])
        npv = sum(CashFlowDiscounter.calc_present_values(taus, cash_flows, 0.0))
        self.assertAlmostEqual(0.0, npv, places=12)
        # case of none-zero discount rate and zero npv
        rate = -0.05
        taus = np.array([0.0, 1.0])
        cash_flows = np.array([-np.exp(-rate * 1.0), 1.0])
        npv = sum(CashFlowDiscounter.calc_present_values(taus, cash_flows, rate))
        self.assertAlmostEqual(0.0, npv, places=12)
        # case of none-zero discount rate and zero npb
        rate = 0.05
        taus = np.array([0.0, 1.0])
        cash_flows = np.array([-np.exp(-rate * 1.0), 1.0])
        npv = sum(CashFlowDiscounter.calc_present_values(taus, cash_flows, rate))
        self.assertAlmostEqual(0.0, npv, places=12)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass

    def test_calc_irr(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        # case of zero rate
        rate = 0.0
        taus = np.array([0.0, 1.0, 2.0])
        cash_flows = np.array([-2.0, 1.0, 1.0])
        cfd = CashFlowDiscounter(taus, cash_flows)
        irr = cfd.calc_irr()
        self.assertAlmostEqual(rate, irr, places=6)
        # case of negative rate
        rate = -0.05
        taus = np.array([0.0, 1.0, 2.0])
        cash_flows = np.array([-2.0, 1.0 * np.exp(rate * 1.0), 1.0 * np.exp(rate * 2.0)])
        cfd = CashFlowDiscounter(taus, cash_flows)
        irr = cfd.calc_irr()
        self.assertAlmostEqual(rate, irr, places=6)
        # case of negative rate
        rate = 0.05
        taus = np.array([0.0, 1.0, 2.0])
        cash_flows = np.array([-2.0, 1.0 * np.exp(rate * 1.0), 1.0 * np.exp(rate * 2.0)])
        cfd = CashFlowDiscounter(taus, cash_flows)
        irr = cfd.calc_irr()
        self.assertAlmostEqual(rate, irr, places=6)
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
