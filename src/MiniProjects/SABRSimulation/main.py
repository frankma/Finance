import matplotlib.pyplot as plt

from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'

fwd = 0.05
tau = 0.75
alpha = 0.4 * fwd
beta = 0.0
nu = 0.88
rho = -0.25

model = SABRModel(tau, alpha, beta, nu, rho)
freq_sim, base_sim = model.sim_forward_den(fwd, n_bins=1000)
freq_lognormal, base_lognormal = model.calc_lognormal_fwd_den(fwd, n_bins=1000)
freq_normal, base_normal = model.calc_normal_fwd_den(fwd, n_bins=1000)

sum_sim = sum(base_sim[:-1] * freq_sim[:-1] * (base_sim[1:] - base_sim[:-1]))
sum_lognormal = sum(base_lognormal[:-1] * freq_lognormal[:-1] * (base_lognormal[1:] - base_lognormal[:-1]))
sum_normal = sum(base_normal[:-1] * freq_normal[:-1] * (base_normal[1:] - base_normal[:-1]))
one_sim = sum(freq_sim[:-1] * (base_sim[1:] - base_sim[:-1]))
one_lognormal = sum(freq_lognormal[:-1] * (base_lognormal[1:] - base_lognormal[:-1]))
one_normal = sum(freq_normal[:-1] * (base_normal[1:] - base_normal[:-1]))

print('\t simulated\t lognormal\t normal\n'
      'mean\t%.12f\t%.12f\t%.12f\n'
      'cumulative\t%.6f\t%.6f\t%.6f'
      % (sum_sim, sum_lognormal, sum_normal,
         one_sim, one_lognormal, one_normal))

plt.plot(base_sim, freq_sim)
plt.plot(base_lognormal, freq_lognormal)
plt.plot(base_normal, freq_normal)
plt.plot([-0.2 * fwd, 5.0 * fwd], [0.0, 0.0], 'k-')
plt.plot([0.0, 0.0], [-0.1 * max(freq_sim), 1.1 * max(freq_sim)], 'k-')
plt.legend(['simulated', 'lognormal', 'normal'])
plt.xlim([-0.2 * fwd, 5.0 * fwd])
plt.ylim([-0.1 * max(freq_sim), 1.1 * max(freq_sim)])
plt.show()
