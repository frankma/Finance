import numpy as np

from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedging import SingleVariableDeltaHedging
from src.MiniProjects.DeltaHedging.SingleVariableDeltaHedgingValuator import SingleVariableDeltaHedgingValuator
from src.Simulator.SingleVariableSimulator import SingleVariableSimulator
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


s_0 = 100.0
k = 100.0
tau = 0.25
opt_type = OptionType.call
r = 0.05
q = 0.0
sig = 0.2
n_scn = 5 * 10**5
n_step = 20
threshold = 0.0

simulator = SingleVariableSimulator(n_scn, s_0, r - q, sig, 'lognormal')
valuator = SingleVariableDeltaHedgingValuator(k, r, q, sig, opt_type)
dh = SingleVariableDeltaHedging(simulator, valuator, n_step, tau, r, threshold)
dh.simulate_to_terminal()
pnl = dh.evaluate()
print(np.average(pnl), np.std(pnl))
dh.graphical_analysis()
