import numpy as np

from src.FDSolver.TriDiagonalMatrix import TriDiagonalMatrix

__author__ = 'frank.ma'


def diffusion_coe(tt):
    return 0.16 + 0.08 * np.sin(tt)


prev = 0.0

print('x incremental\tt incremental\tx value\ty value\tincremental error\tincremental error times 4')

for n_x, n_t in [(11, 9), (21, 17), (41, 33), (81, 65), (161, 129)]:

    xs = np.linspace(0.0, 1.0, num=n_x)
    ts = np.linspace(0.0, 1.0, num=n_t)

    dx = 1.0 / (n_x - 1)
    dt = 1.0 / (n_t - 1.0)
    mu = dx * dx / dt

    state = np.matrix(xs * (np.ones(n_x) - xs)).transpose()
    coe_curr = diffusion_coe(ts[0])

    for idx, t in enumerate(ts[:-1]):
        coe_next = diffusion_coe(t + dt)
        td_curr = TriDiagonalMatrix(0.5 * coe_curr, mu - coe_curr, 0.5 * coe_curr, n_x - 2)
        td_next = TriDiagonalMatrix(-0.5 * coe_next, mu + coe_next, -0.5 * coe_next, n_x - 2)
        coe_curr = coe_next

        state[0] = 0.0
        state[n_x - 1] = 0.0
        state[1:- 1] = td_next.get_inverse() * td_curr.get_matrix() * state[1:- 1]

    diff = state[((n_x - 1) / 2, 0)] - prev
    prev = state[((n_x - 1) / 2, 0)]
    print('%.6f\t%.6f\t%.2f\t%.8f\t%.4e\t%.4e' % (dx, dt, xs[(n_x - 1) / 2], state[((n_x - 1) / 2, 0)], diff, diff / 4))
