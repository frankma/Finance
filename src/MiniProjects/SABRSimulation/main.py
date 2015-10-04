import numpy as np
import matplotlib.pyplot as plt

from src.MiniProjects.SABRSimulation.SABRSimulator import SABRSimulator

__author__ = 'frank.ma'


fwd = 100.0
tau = 0.25
alpha = 0.2
beta = 1.0
nu = 0.7
rho = 0.25

n_scn = 10**5
n_stp = 5 * 10**2

sim = SABRSimulator(fwd, tau, alpha, beta, nu, rho)
strikes = np.linspace(0.1 * fwd, 2.0 * fwd, num=101)
analytic_density = sim.calc_analytic_pdf(strikes)
flat_density = sim.calc_analytic_pdf_given_vols(strikes, alpha)
forwards = sim.simulate(n_stp, n_scn)
plt.hist(forwards, bins=strikes, normed=True)
plt.plot(strikes, analytic_density, 'bx-')
plt.plot(strikes, flat_density, 'r+-')
plt.show()
