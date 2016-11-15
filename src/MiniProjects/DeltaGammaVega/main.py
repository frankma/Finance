import numpy as np
import matplotlib.pyplot as plt

from src.Utils.Valuator.BSM import BSM, BSMVec
from src.Utils.Types.OptionType import OptionType

__author__ = 'frank.ma'

s = 100.
k = 105.
tau = 1.
r = 0.02
q = 0.04
sig = 0.4
opt_type = OptionType.call

price = BSM.price(s, k, tau, r, q, sig, opt_type)
bump = 0.01

s_d = s * (1. - bump)
s_u = s * (1. + bump)
sig_d = sig * (1. - bump)
sig_u = sig * (1. + bump)

p_d_s = BSM.price(s_d, k, tau, r, q, sig, opt_type)
p_u_s = BSM.price(s_u, k, tau, r, q, sig, opt_type)
p_d_v = BSM.price(s, k, tau, r, q, sig_d, opt_type)
p_u_v = BSM.price(s, k, tau, r, q, sig_u, opt_type)

delta_num = (p_u_s - p_d_s) / (2. * s * bump)
delta = BSM.delta(s, k, tau, r, q, sig, opt_type)
gamma_num = (p_u_s - 2. * price + p_d_s) / ((s * bump) ** 2)
gamma = BSM.gamma(s, k, tau, r, q, sig)
vega_num = (p_u_v - p_d_v) / (2. * sig * bump)
vega = BSM.vega(s, k, tau, r, q, sig)

print('Numerical\tAnalytical')
print('%6f\t%6f' % (delta_num, delta))
print('%6f\t%6f' % (gamma_num, gamma))
print('%6f\t%6f' % (vega_num, vega))

s_vec = np.array(np.linspace(0.1 * s, 1.9 * s, num=100))
price_s_vec = BSMVec.price(s_vec, k, tau, r, q, sig, opt_type)
pnl_s = price_s_vec - price
s_ret = s_vec / s - 1.0
pnl_rep_dg = delta_num * s_ret * 100. + 0.5 * gamma_num * s_ret * s_ret * 10000.

sig_vec = np.array(np.linspace(0.1 * sig, 2.0 * sig, num=100))
price_s_vec = BSMVec.price(s, k, tau, r, q, sig_vec, opt_type)
pnl_v = price_s_vec - price
sig_ret = sig_vec / sig - 1.0
pnl_rep_v = vega_num * sig_ret * sig

plt.subplot(2, 2, 1)
plt.plot(s_ret, pnl_s, '.-', label='PnL')
plt.plot(s_ret, pnl_rep_dg, '.-', label='PnL Rep')
plt.plot(s_ret, np.zeros(s_vec.shape))
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(s_ret, pnl_s - pnl_rep_dg, '-', label='residual')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(sig_ret, pnl_v, '.-', label='PnL_v')
plt.plot(sig_ret, pnl_rep_v, '.-', label='PnL Rep Vega')
plt.plot(sig_ret, np.zeros(sig_vec.shape))
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(sig_ret, pnl_v - pnl_rep_v, '-', label='residual')
plt.grid()
plt.show()
