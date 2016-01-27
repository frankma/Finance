import string

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

__author__ = 'frank.ma'

df_ds = pd.read_csv('./SampleDataSet.csv', skiprows=0)
df_ds = df_ds.diff()
df_ds.columns = [s for s in string.ascii_lowercase[:df_ds.columns.__len__()]]
# print(df_f.info)
df_rd = pd.DataFrame(np.random.standard_normal(size=df_ds.shape), index=df_ds.index, columns=df_ds.columns)
mu = df_rd.apply(np.average)
sig = df_rd.apply(np.std)
cov = df_ds.cov()
print(cov)

w, v = np.linalg.eig(cov)
print(w)
print(v)

# colours = cm.spectral(np.linspace(0.0, 1.0, num=10))
#
# plt.figure(1)
# for idx, column in enumerate(df_ds.columns):
#     plt.plot(df_ds.index, df_ds[column], '.', c=colours[idx])
# plt.legend(df_ds.columns)
#
# plt.figure(2)
# for idx, column in enumerate(df_rd.columns):
#     plt.plot(df_rd.index, df_rd[column], '+', c=colours[idx])
# plt.legend(df_rd.columns)
# plt.show()
