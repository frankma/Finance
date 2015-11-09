import numpy as np
import pandas as pd

from src.SolverFD.InitialCondition1D import InitialCondition1D
from src.SolverFD.BoundaryCondition1D import BoundaryCondition1D

__author__ = 'frank.ma'


class FDSolver1D(object):
    def __init__(self, spaces: np.array, times: np.array, init_condition: InitialCondition1D,
                 bound_condition: BoundaryCondition1D, storage: np.array):
        self.spaces = spaces
        self.init_condition = init_condition
        self.bound_condition = bound_condition
        self.state = self.init_condition.get_state(spaces)
        self.idx = 0.0
        self.storage = storage
        self.times = np.unique(np.concatenate((times, storage)))
        self.diagnostic = pd.DataFrame(columns=self.spaces)

    def evolve(self, dt: float):
        
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
