import numpy as np
import matplotlib.pyplot as plt

from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'


fwd = 150.0
tau = 0.5
alpha = 0.4
beta = 1.0
nu = 0.88
rho = 0.0

strikes = np.linspace(0.001 * fwd, 2.0 * fwd, num=100)

model = SABRModel(tau, alpha, beta, nu, rho)
freq_sim, base_sim = model.sim_forward_den(fwd)
freq_ana, base_ana = model.calc_forward_den(fwd)

sum_sim = sum(base_sim[:-1] * freq_sim[:-1] * (base_sim[1:] - base_sim[:-1]))
sum_ana = sum(base_ana[:-1] * freq_ana[:-1] * (base_ana[1:] - base_ana[:-1]))
one_sim = sum(freq_sim[:-1] * (base_sim[1:] - base_sim[:-1]))
one_ana = sum(freq_ana[:-1] * (base_ana[1:] - base_ana[:-1]))

print(sum_sim, sum_ana)
print(one_sim, one_ana)

plt.plot(base_sim, freq_sim, 'bx-')
plt.plot(base_ana, freq_ana, 'r+-')
plt.legend(['simulated', 'analyzed'])
plt.xlim([strikes[0], strikes[-1]])
plt.ylim([-0.1 * max(freq_sim), 1.1 * max(freq_sim)])
plt.show()
