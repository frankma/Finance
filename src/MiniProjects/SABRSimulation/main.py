import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from src.MiniProjects.SABRSimulation.SABRSimulator import SABRSimulator
from src.SABRModel.SABRModel import SABRModel
from src.Utils.Black76 import Black76
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


fwd = 150.0
tau = 0.5
alpha = 0.4
beta = 1.0
nu = 0.88
rho = 0.0

n_scn = 10**6
n_stp = 10**2

strikes = np.linspace(0.2 * fwd, 4.0 * fwd, num=201)
strikes_mid = 0.5 * (strikes[:-1] + strikes[1:])
strikes_inc = np.subtract(strikes[1:], strikes[:-1])

sim = SABRSimulator(fwd, tau, alpha, beta, nu, rho)

sim_fwds_terminal = sim.simulate(n_stp, n_scn)

var_vols = sim.calc_sigmas(strikes_mid)
var_den = sim.calc_analytic_pdf(strikes_mid)
flat_vols = sim.calc_sigmas(np.array([fwd]))
flat_den = sim.calc_pdf_given_vols(strikes_mid, flat_vols)

sim_mean = np.average(sim_fwds_terminal)
sim_std = np.std(sim_fwds_terminal)
var_mean = sum(strikes_mid * var_den * strikes_inc)
var_std = sqrt(sum(np.subtract(strikes_mid, var_mean)**2 * var_den * strikes_inc))
flat_mean = sum(strikes_mid * flat_den * strikes_inc)
flat_std = sqrt(sum(np.subtract(strikes_mid, flat_mean)**2 * flat_den * strikes_inc))

print('\tmean\tstd')
print('sim\t%.6f\t%6f' % (sim_mean, sim_std))
print('var vol\t%6f\t%6f\t%6f' % (var_mean, var_std, sum(var_den * strikes_inc)))
print('flat vol\t%6f\t%6f\t%6f' % (flat_mean, flat_std, sum(flat_den * strikes_inc)))

print('option pricing\nsimulated\tanalytic')
den_y, bins = np.histogram(sim_fwds_terminal, bins=strikes, normed=True)
den_x = (bins[1:] + bins[:-1]) * 0.5

k = 155.0
b = 0.95
opt_type = OptionType.call

payoff = np.maximum(opt_type.value * np.subtract(den_x, k), np.zeros(den_y.__len__()))
den_inc = np.subtract(bins[1:], bins[:-1])
simulated_price = b * sum(payoff * den_y * den_inc)

black_vol = SABRModel(tau, alpha, beta, nu, rho).calc_lognormal_vol(fwd, k)
black_price = Black76.price(fwd, k, tau, black_vol, b, opt_type)

print('%.6f\t%.6f' % (simulated_price, black_price))

freq, base = SABRModel(tau, alpha, beta, nu, rho).sim_forward_den(fwd)

plt.subplot(3, 1, 1)
plt.plot(strikes_mid, var_vols, 'b')
plt.xlim([strikes[0], strikes[-1]])
plt.subplot(3, 1, 2)
plt.hist(sim_fwds_terminal, bins=strikes, normed=True, color='w')
plt.plot(strikes_mid, var_den, 'b')
plt.plot(strikes_mid, flat_den, 'r')
plt.xlim([strikes[0], strikes[-1]])
plt.subplot(3, 1, 3)
plt.plot(base, freq)
plt.xlim([strikes[0], strikes[-1]])
plt.show()
