import time as tm

import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis

from src.MiniProjects.DeltaHedging.SingleVarDeltaHedging import SingleVarDeltaHedging
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


N_SCN = 10**5
N_STPS = [48, 92, 184]
S_0 = 100
K = 100
TAU = 0.25
R_SIM = 0.0
Q_SIM = 0.0
SIG_SIMS = [0.3, 0.5]
R_OPT = R_SIM
Q_OPT = Q_SIM
OPT_TYPE = OptionType.put

bins = np.linspace(-8.0, 8.0, 41)
bins_mid = 0.5 * (bins[1:] + bins[:-1])

df_pnl = pd.DataFrame()
df_hist = pd.DataFrame()

print('trail_id\tmean\tstd_dev\tskew\tcalc_time')
for SIG_SIM in SIG_SIMS:
    for N_STP in N_STPS:
        trail_id = SIG_SIM.__str__() + '_' + N_STP.__str__()
        SIG_OPT = SIG_SIM
        tic = tm.time()
        dh = SingleVarDeltaHedging(N_SCN, N_STP, S_0, K, TAU, R_SIM, Q_SIM, SIG_SIM, R_OPT, Q_OPT, SIG_OPT, OPT_TYPE)
        p_n_l, c_a, pff = dh.sim_to_term()
        print('%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f'
              % (trail_id, np.average(p_n_l), np.std(p_n_l), skew(p_n_l), kurtosis(p_n_l), (tm.time() - tic)))
        df_pnl = pd.concat([df_pnl, pd.DataFrame(data=p_n_l, columns=[trail_id])], axis=1, join='inner')
        # histogram analysis
        freq, bins_ret = np.histogram(p_n_l, bins)
        df_hist = pd.concat([df_hist, pd.DataFrame(data=freq, index=bins_mid, columns=[trail_id])], axis=1, join='inner')

# print(df_pnl.info())
print(df_hist)