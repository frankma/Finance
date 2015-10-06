import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from src.MiniProjects.SABRSimulation.SABRSimulator import SABRSimulator

__author__ = 'frank.ma'


fwd = 150.0
tau = 0.5
alpha = 0.4
beta = 1.0
nu = 0.88
rho = 0.0

n_scn = 10**6
n_stp = 10**2

strikes = np.linspace(0.2 * fwd, 2.0 * fwd, num=101)
strikes_mid = 0.5 * (strikes[:-1] + strikes[1:])
strikes_ana = np.linspace(0.1 * fwd, 10.0 * fwd, num=1001)
strikes_ana_mid = 0.5 * (strikes_ana[:-1] + strikes_ana[1:])
strikes_ana_inc = np.subtract(strikes_ana[1:], strikes_ana[:-1])

sim = SABRSimulator(fwd, tau, alpha, beta, nu, rho)

sigma_atm = sim.calc_sigmas(np.array([fwd]))
sigmas = sim.calc_sigmas(strikes_ana_mid)
var_den = sim.calc_analytic_pdf(strikes_ana_mid)
flat_den = sim.calc_pdf_given_vols(strikes_ana_mid, sigma_atm)
forwards = sim.simulate(n_stp, n_scn)

sim_mean = np.average(forwards)
sim_std = np.std(forwards)
var_mean = sum(strikes_ana_mid * var_den * strikes_ana_inc)
var_std = sqrt(sum(np.subtract(strikes_ana_mid, var_mean)**2 * var_den * strikes_ana_inc))
flat_mean = sum(strikes_ana_mid * flat_den * strikes_ana_inc)
flat_std = sqrt(sum(np.subtract(strikes_ana_mid, flat_mean)**2 * flat_den * strikes_ana_inc))

print('\tmean\tstd')
print('sim\t%.6f\t%6f' % (sim_mean, sim_std))
print('var vol\t%6f\t%6f\t%6f' % (var_mean, var_std, sum(var_den * strikes_ana_inc)))
print('flat vol\t%6f\t%6f\t%6f' % (flat_mean, flat_std, sum(flat_den * strikes_ana_inc)))

plt.subplot(2, 1, 1)
plt.plot(strikes_ana_mid, sigmas, 'b')
plt.xlim([strikes[0], strikes[-1]])
plt.subplot(2, 1, 2)
plt.hist(forwards, bins=strikes, normed=True, color='w')
plt.plot(strikes_ana_mid, var_den, 'b')
plt.plot(strikes_ana_mid, flat_den, 'r')
plt.xlim([strikes[0], strikes[-1]])
plt.show()
