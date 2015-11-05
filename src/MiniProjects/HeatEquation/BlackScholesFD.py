import numpy as np
from src.SolverFD.TriDiagonalMatrix import TriDiagonalMatrix
from src.Utils.BSM import BSM, BSMVecS
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BlackScholesFD(object):
    def __init__(self, t: float, s: float, k: float, r: float, sig: float, n_x: int = 100, n_t: int = 100,
                 domain: float = 8.0):
        self.n_x = n_x
        self.n_t = n_t
        self.k = k
        self.s = s
        self.r = r
        self.sig = sig
        self.xs = np.linspace(1e-12, domain * sig * max(k, s), num=n_x + 1)
        self.ts = np.linspace(t, 0.0, num=n_t + 1)
        self.dx = self.xs[1] - self.xs[0]
        self.dt = self.ts[0] - self.ts[1]
        self.sdx = self.xs / self.dx
        self.tdx = self.ts / self.dt

    def solve(self):
        opt_type = OptionType.call
        state = np.matrix(np.maximum(opt_type.value * (self.xs - self.k), np.full(self.xs.__len__(), 0.0)))
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

        for t in self.ts[:-1]:
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

    # for n_t in [16, 32, 64, 128, 256]:
    for n_x in [256, 512, 1024, 2048, 4096]:
        prev = 0.0
        # for n_x in [256, 512, 1024, 2048, 4096]:
        for n_t in [32, 64, 128, 256, 512]:
            bsm_fd = BlackScholesFD(tau, spot, strike, rate, vol, n_x=n_x, n_t=n_t)
            res = bsm_fd.solve()
            print('%i\t%i\t%.16f\t%.2e\t%.2e' % (n_x, n_t, res, res - prev, res - ana))
            prev = res
