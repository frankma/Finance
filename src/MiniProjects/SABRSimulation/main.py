import numpy as np
import matplotlib.pyplot as plt

from src.MiniProjects.SABRSimulation.SABRSimulator import SABRSimulator

__author__ = 'frank.ma'


fwd = 150.0
tau = 0.5
alpha = 0.4
beta = 1.0
nu = 0.9
rho = -0.4

n_scn = 10**6
n_stp = 10**2

strikes = np.linspace(0.2 * fwd, 2.0 * fwd, num=101)
sim = SABRSimulator(fwd, tau, alpha, beta, nu, rho)

sigmas = sim.calc_sigmas(strikes)
analytic_density = sim.calc_analytic_pdf(strikes)
flat_density = sim.calc_analytic_pdf_given_vols(strikes, sim.calc_sigmas(np.array([fwd])))
forwards = sim.simulate(n_stp, n_scn)

plt.subplot(2, 1, 1)
plt.plot(strikes, sigmas, 'bx-')
plt.xlim([strikes[0], strikes[-1]])
plt.subplot(2, 1, 2)
plt.hist(forwards, bins=strikes, normed=True, color='m')
plt.plot(strikes, analytic_density, 'bx-')
plt.plot(strikes, flat_density, 'r+-')
plt.xlim([strikes[0], strikes[-1]])
plt.show()
