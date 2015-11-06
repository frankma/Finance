import numpy as np

from src.SolverFD.InitialState1D import InitialState1D
from src.SolverFD.BoundaryCondition1D import BoundaryCondition1D

__author__ = 'frank.ma'


class FDSolver1D(object):
    def __init__(self, spaces: np.array, times: np.array, init_state: InitialState1D, bound_cond: BoundaryCondition1D):
        self.spaces = spaces
        self.times = times
        self.init_state = init_state
        self.bound_cond = bound_cond
        self.state = init_state.get_init_state(spaces)
        pass

    def solve(self):
        pass

    def index(self, xs: float or np.array) -> float or np.array:
        if isinstance(xs, float):
            return np.interp([xs], self.spaces, self.state)[0]
        elif isinstance(xs, np.ndarray):
            return np.interp(xs, self.spaces, self.state)
        else:
            raise ValueError('unrecognized type (%s) of xs, expect float or np.array' % type(xs))
