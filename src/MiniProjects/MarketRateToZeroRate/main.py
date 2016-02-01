import numpy as np

from src.MiniProjects.MarketRateToZeroRate.MarketRateToZeroRate import MarketRateToZeroRate
from src.Utils.Valuator.CashFlowDiscounter import CashFlowDiscounter

__author__ = 'frank.ma'

ts_1 = np.array([0.0, 0.5, 1.0, 1.0])
cs_1 = np.array([-1.0, 0.05, 0.05, 1.0])
bond_1 = CashFlowDiscounter(ts_1, cs_1)

ts_2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.0])
cs_2 = np.array([-1.0, 0.06, 0.06, 0.06, 0.06, 1.0])
bond_2 = CashFlowDiscounter(ts_2, cs_2)

ts_3 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
cs_3 = np.array([-1.0, 0.12, 0.12, 0.12, 0.12, 0.12, 1.0])
bond_3 = CashFlowDiscounter(ts_3, cs_3)


bonds = [bond_1, bond_2, bond_3]
weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
tenors = np.array([0.999, 2.0, 5.0001])

rate_converter = MarketRateToZeroRate(bonds, weights, tenors)
market_rates = rate_converter.market_rates
zero_rates = rate_converter.fit_zero_curve()

print(market_rates)
print(zero_rates)