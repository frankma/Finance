import numpy as np
from src.FDSolver.TriDiagonalMatrix import TriDiagonalMatrix
from src.Utils.BSM import BSM
from src.Utils.OptionType import OptionType

__author__ = 'frank.ma'


class BlackScholesFD(object):
    def __init__(self, t: float, s: float, k: float, r: float, sig: float, n_x: int = 100, n_t: int = 100,
                 domain: float = 4.0):
        self.n_x = n_x
        self.n_t = n_t
        self.k = k
        self.s = s
        self.r = r
        self.sig = sig
        self.dx = 1.0 / n_x
        self.dt = 1.0 / n_t
        self.ts = np.linspace(0.0, t, num=n_t)
        self.xs = np.linspace(1e-6, domain * k, num=n_x)

    def solve(self):
        state = np.matrix(np.maximum(self.xs - self.k, np.zeros(self.n_x, dtype=float))).transpose()
        lhs_l = (((self.sig * self.xs) ** 2) - self.r * self.xs * self.dx) / 4.0
        lhs_c = (self.dx ** 2) / self.dt - (((self.sig * self.xs) ** 2) + self.r * (self.dx ** 2)) / 2.0
        lhs_u = (((self.sig * self.xs) ** 2) + self.r * self.xs * self.dx) / 4.0
        lhs_tdm = TriDiagonalMatrix(lhs_l[1:], lhs_c, lhs_u[:-1])
        rhs_l = -lhs_l
        rhs_c = (self.dx ** 2) / self.dt + (((self.sig * self.xs) ** 2) + self.r * (self.dx ** 2)) / 2.0
        rhs_u = -lhs_u
        rhs_tdm = TriDiagonalMatrix(rhs_l[1:], rhs_c, rhs_u[:-1])

        for t in self.ts[:-1]:
            print(state.transpose())
            state = lhs_tdm.get_inverse() * rhs_tdm.get_matrix() * state

        vec = np.asarray(state.transpose()).reshape(-1)
        # print(vec)
        value = np.interp([self.s], self.xs, vec)[0]
        print(value)


if __name__ == '__main__':
    tau = 0.75
    spot = 150.0
    strike = 155.0
    rate = 0.01
    vol = 0.25

    bsm_fd = BlackScholesFD(tau, spot, strike, rate, vol, n_x=11, n_t=11)
    bsm_fd.solve()

    ana = BSM.price(spot, strike, tau, rate, 0.0, vol, OptionType.call)
    print(ana)
