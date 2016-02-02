import numpy as np

from src.MiniProjects.MarketRateToZeroRate.MarketRateToZeroRate import MarketRateToZeroRate
from src.Utils.Valuator.CashFlowDiscounter import CashFlowDiscounter
import matplotlib.pyplot as plt

__author__ = 'frank.ma'

ts_1 = np.array([0.0, 0.5, 1.0, 1.0])
cs_1 = np.array([-1.005, 0.003, 0.003, 1.0])
bond_1 = CashFlowDiscounter(ts_1, cs_1)

ts_2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.0])
cs_2 = np.array([-0.98, 0.005, 0.005, 0.005, 0.015, 1.0])
bond_2 = CashFlowDiscounter(ts_2, cs_2)

ts_3 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
cs_3 = np.array([-0.975, 0.03, 0.03, 0.03, 0.03, 0.03, 1.0])
bond_3 = CashFlowDiscounter(ts_3, cs_3)

ts_4 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0])
cs_4 = np.array([-0.96, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0])
bond_4 = CashFlowDiscounter(ts_4, cs_4)

ts_5 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0])
cs_5 = np.array([-1.0, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 1.0])
bond_5 = CashFlowDiscounter(ts_5, cs_5)

ts_6 = np.array(list(range(0, 21)) + [20])
cs_6 = np.append(np.append([-0.98], np.full(ts_6.__len__() - 2, 0.06)), [1.0])
bond_6 = CashFlowDiscounter(ts_6, cs_6)

bonds = [bond_1, bond_2, bond_3, bond_4, bond_5, bond_6]
weights = [1.0 / bonds.__len__() for _ in range(bonds.__len__())]
tenors = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

rate_converter = MarketRateToZeroRate(bonds, weights, tenors)
market_rates = rate_converter.market_rates
zero_rates = rate_converter.fit_zero_curve()

print(market_rates)
print(zero_rates)

plt.plot(rate_converter.bonds_maturities, rate_converter.market_rates, '-+r')
plt.plot(tenors, zero_rates, '-xb')
plt.legend(['mkt rate', 'zero rate'], loc=4)
plt.xlim([0.0, 25.0])
plt.ylim([0.0, 0.08])
plt.show()
