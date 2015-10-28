import numpy as np
from src.MiniProjects.HeatEquation.TriDiagonalMatrix import TriDiagonalMatrix

__author__ = 'frank.ma'


def diffusion_coe(tt):
    return 0.16 + 0.08 * np.sin(tt)


n_t = 101
n_x = 321

ts = np.linspace(0.0, 1.0, num=n_t)
xs = np.linspace(0.0, 1.0, num=n_x)

dt = ts[1] - ts[0]
dx = xs[1] - xs[0]
mu = (dx ** 2) / dt

state = np.matrix(xs * (np.ones(n_x) - xs)).transpose()
coe_prev = diffusion_coe(ts[0])

for idx, t in enumerate(ts[1:]):
    td_prev = TriDiagonalMatrix(0.5 * coe_prev, mu - coe_prev, 0.5 * coe_prev, n_x)
    coe_curr = diffusion_coe(t)
    td_curr = TriDiagonalMatrix(-0.5 * coe_curr, mu + coe_curr, -0.5 * coe_curr, n_x)
    coe_prev = coe_curr
    state[0] = 0.0
    state[n_x - 1] = 0.0
    state = td_curr.get_inverse() * (td_prev.get_matrix() * state)

print(state[(n_x - 1) / 2])
