import numpy as np

from src.SolverFD.TriDiagonalMatrix import TriDiagonalMatrix
from src.Utils.Types.OptionType import OptionType
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'


class BlackScholesBackwardPDE(object):
    def __init__(self, t: float, s: float, k: float, r: float, sig: float, n_x: int = 100, n_t: int = 100,
                 domain: float = 10.0):
        self.n_x = n_x
        self.n_t = n_t
        self.k = k
        self.s = s
        self.r = r
        self.sig = sig
        self.xs = np.array(np.linspace(1e-12, domain * sig * np.sqrt(t) * max(k, s), num=n_x + 1))
        self.ts = np.linspace(t, 0.0, num=n_t + 1)
        self.ds = self.xs[1] - self.xs[0]
        self.dt = self.ts[0] - self.ts[1]
        self.sdx = self.xs / self.ds
        self.tdx = self.ts / self.dt

    def solve(self):
        state = np.matrix(np.maximum(OptionType.call.value * (self.xs - self.k), np.full(np.shape(self.xs), 0.0)))

        lhs_lft = self.r * self.sdx * self.dt / 4.0 - ((self.sig * self.sdx) ** 2) * self.dt / 4.0
        lhs_ctr = 1.0 + ((self.sig * self.sdx) ** 2) * self.dt / 2.0 + self.r * self.dt / 2.0
        lhs_upr = -self.r * self.sdx * self.dt / 4.0 - ((self.sig * self.sdx) ** 2) * self.dt / 4.0
        lhs_ctr[0] = 1.0 + self.r * self.sdx[0] * self.dt / 2.0 + self.r * self.dt / 2.0
        lhs_upr[0] = -self.r * self.sdx[0] * self.dt / 2.0
        lhs_lft[-1] = self.r * self.sdx[-1] * self.dt / 2.0
        lhs_ctr[-1] = 1.0 - self.r * self.sdx[-1] * self.dt / 2.0 + self.r * self.dt / 2.0
        # lhs_ctr[0] = 2.0
        # lhs_upr[0] = -1.0
        # lhs_lft[-1] = -1.0
        # lhs_ctr[-1] = 2.0
        lhs_tdm = TriDiagonalMatrix(lhs_lft[1:], lhs_ctr, lhs_upr[:-1])

        rhs_lft = -lhs_lft
        rhs_ctr = 1.0 - ((self.sig * self.sdx) ** 2) * self.dt / 2.0 - self.r * self.dt / 2.0
        rhs_upr = -lhs_upr
        rhs_ctr[0] = 1.0 - self.r * self.sdx[0] * self.dt / 2.0 - self.r * self.dt / 2.0
        rhs_upr[0] = self.r * self.sdx[0] * self.dt / 2.0
        rhs_lft[-1] = -self.r * self.sdx[-1] * self.dt / 2.0
        rhs_ctr[-1] = 1.0 + self.r * self.sdx[-1] * self.dt / 2.0 - self.r * self.dt / 2.0
        # rhs_ctr[0] = 2.0
        # rhs_upr[0] = -1.0
        # rhs_lft[-1] = -1.0
        # rhs_ctr[-1] = 2.0
        rhs_tdm = TriDiagonalMatrix(rhs_lft[1:], rhs_ctr, rhs_upr[:-1])

        tm = rhs_tdm.get_matrix().transpose() * lhs_tdm.get_inverse().transpose()

        for _ in self.ts[:-1]:
            state *= tm  # save vector space for calculation

        vec = np.asarray(state).reshape(-1)
        return np.interp([self.s], self.xs, vec)[0]


if __name__ == '__main__':
    tau = 0.5
    spot = 150.0
    strike = 150.0
    rate = 0.05
    vol = 0.45

    ana = BSM.price(spot, strike, tau, rate, 0.0, vol, OptionType.call)
    print(ana)

    # for n_t in [16, 32, 64, 128]:
    for num_s in [256, 512, 1024, 2048]:
        prev = 0.0
        # for n_s in [256, 512, 1024, 2048, 4096]:
        for num_t in [16, 32, 64, 128]:
            bsm_fd = BlackScholesBackwardPDE(tau, spot, strike, rate, vol, n_x=num_s, n_t=num_t)
            res = bsm_fd.solve()
            print('%i\t%i\t%.16f\t%.2e\t%.2e' % (num_s, num_t, res, res - prev, res - ana))
            prev = res
