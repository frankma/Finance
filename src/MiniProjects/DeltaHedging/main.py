import numpy as np

from src.MiniProjects.DeltaHedging.SingleVarDeltaHedging import SingleVarDeltaHedging
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'

N_SCN = 10**5
N_STP = 120
S_0 = 100
K = 100
TAU = 0.25
R_SIM = 0.0
Q_SIM = 0.0
SIG_SIM = 0.2
R_OPT = R_SIM
Q_OPT = Q_SIM
SIG_OPT = SIG_SIM
OPT_TYPE = OptionType.call

dh = SingleVarDeltaHedging(N_SCN, N_STP, S_0, K, TAU, R_SIM, Q_SIM, SIG_SIM, R_OPT, Q_OPT, SIG_OPT, OPT_TYPE, 'lognormal')
p_n_l, c_a, pff = dh.sim_to_terminal()
# print(p_n_l, c_a, pff)
print(np.average(p_n_l), np.std(p_n_l))
