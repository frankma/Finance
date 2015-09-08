from math import log, sqrt


__author__ = 'frank.ma'


class SABRModel(object):

    def __init__(self, t: float, alpha: float, beta: float, nu: float, rho: float):
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.rho = rho

    def calc_lognormal_vol(self, f: float, k: float):
        one_m_beta = 1.0 - self.beta
        f_mul_k = f * k
        f_per_k = f / k

        term1, term2, term3 = 1.0, 1.0, 1.0
        if abs(k - f) < 1e-10:
            # At-The-Money option reduces term1 and term2 as log(f / k_atm) is zero
            term1 = self.alpha / f**one_m_beta
        else:
            z = self.nu / self.alpha * f_mul_k**(one_m_beta / 2.0) * log(f_per_k)
            x = log((sqrt(1.0 - 2.0 * self.rho * z + z**2) + z - self.rho) / (1.0 - self.rho))
            term1 = self.alpha / (f_mul_k**(one_m_beta / 2.0) * (1.0 + one_m_beta**2 / 24.0 * (log(f_per_k))**2 +
                                                                 one_m_beta**4 / 1920.0 * (log(f_per_k))**4))
            term2 = z / x
        term3 = (1.0 + (one_m_beta**2 / 24.0 * self.alpha**2 / (f_mul_k**one_m_beta) +
                        0.25 * self.rho * self.beta * self.nu * self.alpha / f_mul_k**(one_m_beta / 2.0) +
                        (2.0 - 3.0 * self.rho**2) / 24.0 * self.nu**2) * self.t)

        return term1 * term2 * term3

    def calc_normal_vol(self, f: float, k: float):
        pass
