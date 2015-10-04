import numpy as np
from src.SABRModel.SABRModel import SABRModel

__author__ = 'frank.ma'


class SABRSimulator(object):

    def __init__(self, forward: float, tau: float, alpha: float, beta: float, nu: float, rho: float):
        self.forward_0 = forward
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho
        if not 1.0 >= self.rho >= -1.0:
            raise ValueError('correlation %r must in range [-1.0, 1.0]' % self.rho)

    def calc_analytic_pdf(self, strikes: np.array) -> np.array:
        model = SABRModel(self.tau, self.alpha, self.beta, self.nu, self.rho)
        vols = model.calc_lognormal_vol_vec_k(self.forward_0, strikes)
        return self.calc_analytic_pdf_given_vols(strikes, vols)

    def calc_analytic_pdf_given_vols(self, strikes: np.array, vols: np.array) -> np.array:
        means = np.log(self.forward_0)  # we know sabr is driftless as it models forward
        sigmas = vols * np.sqrt(self.tau)  # give volatility time scale
        density = 1.0 / sigmas / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * ((np.log(strikes) - means) / sigmas)**2)
        return density / self.forward_0

    def calc_mc_pdf(self, strikes: np.array, n_steps: int = 100, n_scenarios: int = 10**5):
        forwards = self.simulate(n_steps, n_scenarios)
        freq, bins = np.histogram(forwards, bins=strikes, normed=True)
        bins_mid = 0.5 * (bins[:-1] + bins[1:])
        return bins_mid, freq

    def simulate(self, n_steps: int, n_scenarios: int) -> np.array:
        taus = np.linspace(self.tau, 0.0, num=n_steps)
        # initial setting up forward and sigma
        forwards = np.full(n_scenarios, self.forward_0)
        sigmas = np.full(n_scenarios, self.alpha)

        for idx, tau in enumerate(taus[1:]):
            sqrt_dt = np.sqrt(taus[idx] - tau)
            # standard random variables with mean of zero, variance of one and correlation as given
            means = [0.0, 0.0]
            covariances = [[1.0, self.rho], [self.rho, 1.0]]
            rands = np.random.multivariate_normal(means, covariances, n_scenarios)
            forwards += sigmas * forwards**self.beta * rands[:, 0] * sqrt_dt
            sigmas += self.nu * sigmas * rands[:, 1] * sqrt_dt

        return forwards
