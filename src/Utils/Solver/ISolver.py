import logging

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class ISolver(object):
    ABS_TOL = 1e-12
    REL_TOL = 1e-6
    ITR_TOL = 999

    def solve(self):
        pass

    def set_abs_tol(self, abs_tol: float):
        self.ABS_TOL = abs_tol

    def set_rel_tol(self, rel_tol: float):
        self.REL_TOL = rel_tol

    def set_itr_tol(self, itr_tol: int):
        self.ITR_TOL = itr_tol
