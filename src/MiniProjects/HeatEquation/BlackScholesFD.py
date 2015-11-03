import numpy as np
from src.FDSolver.TriDiagonalMatrix import TriDiagonalMatrix
from src.Utils.BSM import BSM, BSMVecS
from src.Utils.OptionType import OptionType
import matplotlib.pyplot as plt

__author__ = 'frank.ma'


class BlackScholesFD(object):
    def __init__(self, t: float, s: float, k: float, r: float, sig: float, n_x: int = 100, n_t: int = 100,
                 domain: float = 10.0):
        self.n_x = n_x
        self.n_t = n_t
        self.k = k
        self.s = s
        self.r = r
        self.sig = sig
        self.xs = np.linspace(1e-6, domain * sig * max(k, s), num=n_x)
        self.ts = np.linspace(t, 0.0, num=n_t)
        self.dx = self.xs[1] - self.xs[0]
        self.dt = self.ts[0] - self.ts[1]
        self.sdx = self.xs / self.dx
        self.tdx = self.ts / self.dt

    def solve(self):
        opt_type = OptionType.call
        state = np.matrix(np.maximum(opt_type.value * (self.xs - self.k), np.zeros(self.n_x, dtype=float)))
        lhs_lft = self.r * self.sdx * self.dt / 4.0 - ((self.sig * self.sdx) ** 2) * self.dt / 4.0
        lhs_ctr = 1.0 + ((self.sig * self.sdx) ** 2) * self.dt / 2.0 + self.r * self.dt / 2.0
        lhs_upr = -self.r * self.sdx * self.dt / 4.0 - ((self.sig * self.sdx) ** 2) * self.dt / 4.0
        lhs_ctr[0] = 1.0 + self.r * self.dt / 2.0 + self.r * self.sdx[0] * self.dt / 2.0
        lhs_upr[0] = -self.r * self.sdx[0] * self.dt / 2.0
        lhs_lft[-1] = self.r * self.sdx[-1] * self.dt / 2.0
        lhs_ctr[-1] = 1.0 + self.r * self.dt / 2.0 - self.r * self.sdx[-1] * self.dt / 2.0
        lhs_tdm = TriDiagonalMatrix(lhs_lft[1:], lhs_ctr, lhs_upr[:-1])
        rhs_lft = -lhs_lft
        rhs_ctr = 1.0 - ((self.sig * self.sdx) ** 2) * self.dt / 2.0 - self.r * self.dt / 2.0
        rhs_upr = -lhs_upr
        rhs_ctr[0] = 1.0 - self.r * self.dt / 2.0 - self.r * self.sdx[0] * self.dt / 2.0
        rhs_upr[0] = self.r * self.sdx[0] * self.dt / 2.0
        rhs_lft[-1] = -self.r * self.sdx[-1] * self.dt / 2.0
        rhs_ctr[-1] = 1.0 - self.r * self.dt / 2.0 + self.r * self.sdx[-1] * self.dt / 2.0
        rhs_tdm = TriDiagonalMatrix(rhs_lft[1:], rhs_ctr, rhs_upr[:-1])

        tm = rhs_tdm.get_matrix().transpose() * lhs_tdm.get_inverse().transpose()

        show_idx = [0, 1, int(0.2 * self.n_t), int(0.5 * self.n_t), int(0.75 * self.n_t)]
        for idx, t in enumerate(self.ts[:-1]):
            if idx in show_idx:
                plt.plot(self.xs, state.transpose())
            state *= tm  # save vector space for calculation
        plt.plot(self.xs, state.transpose())
        ana_res = BSMVecS.price(self.xs, self.k, self.ts[0], self.r, 0.0, self.sig, opt_type)
        plt.plot(self.xs, ana_res)
        plt.legend(list(self.ts[np.array(show_idx + [self.n_t - 1])]) + ['analytic'], loc=2)
        plt.xlim([0.0, 2.0 * self.k])
        plt.ylim([0.0, 2.0 * self.k - self.s])
        plt.show()

        vec = np.asarray(state.transpose()).reshape(-1)
        # print(vec)
        value = np.interp([self.s], self.xs, vec)[0]
        print(value)


if __name__ == '__main__':
    tau = 0.5
    spot = 150.0
    strike = 155.0
    rate = 0.05
    vol = 0.45

    bsm_fd = BlackScholesFD(tau, spot, strike, rate, vol, n_x=101, n_t=101)
    bsm_fd.solve()

    ana = BSM.price(spot, strike, tau, rate, 0.0, vol, OptionType.call)
    print(ana)
