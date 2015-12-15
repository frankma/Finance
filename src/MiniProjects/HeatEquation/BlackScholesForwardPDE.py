import numpy as np

from src.SolverFD.TriDiagonalMatrix import TriDiagonalMatrix
from src.Utils.OptionType import OptionType
from src.Utils.Valuator.BSM import BSM

__author__ = 'frank.ma'


class BlackScholesForwardPDE(object):
    def __init__(self, t: float, s: float, k: float, r: float, sig: float, n_x: int = 100, n_t: int = 100,
                 domain: float = 10.0):
        self.n_x = n_x
        self.n_t = n_t
        self.k = k
        self.s = s
        self.r = r
        self.sig = sig
        self.ks = np.array(np.linspace(1e-12, domain * sig * np.sqrt(t) * max(k, s), num=n_x + 1))  # discretize strike
        self.ts = np.linspace(0.0, t, num=n_t + 1)  # forward PDE starts from zero ends at T
        self.taus = self.ts[::-1]
        self.dk = self.ks[1] - self.ks[0]
        self.dt = self.ts[1] - self.ts[0]  # forward pde
        self.kdx = self.ks / self.dk
        self.tdx = self.ts / self.dt

    def solve(self):
        state = np.matrix(np.maximum(OptionType.call.value * (self.s - self.ks), np.full(np.shape(self.ks), 0.0)))

        vol_loc = self.sig  # assume flat vol across time and strike, then local vol equals to implied vol

        lhs_lft = -self.r * self.kdx * self.dt / 4.0 - ((vol_loc * self.kdx) ** 2) * self.dt / 4.0
        lhs_ctr = 1.0 + ((vol_loc * self.kdx) ** 2) * self.dt / 2.0
        lhs_upr = self.r * self.kdx * self.dt / 4.0 - ((vol_loc * self.kdx) ** 2) * self.dt / 4.0
        lhs_ctr[0] = 1.0 - self.r * self.kdx[0] * self.dt / 2.0
        lhs_upr[0] = -self.r * self.kdx[0] * self.dt / 2.0
        lhs_lft[-1] = self.r * self.kdx[-1] * self.dt / 2.0
        lhs_ctr[-1] = 1.0 + self.r * self.kdx[-1] * self.dt / 2.0
        lhs_tdm = TriDiagonalMatrix(lhs_lft[1:], lhs_ctr, lhs_upr[:-1])

        rhs_lft = -lhs_lft
        rhs_ctr = 1.0 - ((vol_loc * self.kdx) ** 2) * self.dt / 2.0
        rhs_upr = -lhs_upr
        rhs_ctr[0] = 1.0 + self.r * self.kdx[0] * self.dt / 2.0
        rhs_upr[0] = self.r * self.kdx[0] * self.dt / 2.0
        rhs_lft[-1] = -self.r * self.kdx[-1] * self.dt / 2.0
        rhs_ctr[-1] = 1.0 - self.r * self.kdx[-1] * self.dt / 2.0
        rhs_tdm = TriDiagonalMatrix(rhs_lft[1:], rhs_ctr, rhs_upr[:-1])

        tm = rhs_tdm.get_matrix().transpose() * lhs_tdm.get_inverse().transpose()

        for _ in self.ts[:-1]:
            state *= tm  # save vector space for calculation

        vec = np.asarray(state).reshape(-1)
        return np.interp([self.k], self.ks, vec)[0]

if __name__ == '__main__':
    tau = 0.5
    spot = 150.0
    strike = 150.0
    rate = 0.05
    div = 0.0  # this has to be zero as carry rate wasn't factored into the later code
    vol = 0.45

    opt_type = OptionType.call
    value = BSM.price(spot, strike, tau, rate, div, vol, opt_type)
    print(value)

    # for n_t in [16, 32, 64, 128]:
    for num_k in [256, 512, 1024, 2048]:
        prev = 0.0
        # for n_k in [256, 512, 1024, 2048, 4096]:
        for num_t in [16, 32, 64, 128]:
            bsm_fd = BlackScholesForwardPDE(tau, spot, strike, rate, vol, n_x=num_k, n_t=num_t)
            res = bsm_fd.solve()
            print('%i\t%i\t%.16f\t%.2e\t%.2e' % (num_k, num_t, res, res - prev, res - value))
            prev = res
