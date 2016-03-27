import logging

import numpy as np

from src.MiniProjects.Sudoku.Sudoku import Sudoku

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SudokuSolver(object):
    def __init__(self,
                 problem: Sudoku):
        self.problem = problem
        self.found_solution = False
        self.solution = np.zeros(problem.sudoku.shape)

    def solve(self):
        pass

    def print_solution(self,
                       output: str,
                       print_partial=False):
        if self.found_solution or print_partial:
            np.savetxt(output, self.solution, fmt='%i', delimiter=' ')
            logger.info('Solution is saved in file %s.' % output)
        else:
            logger.warning('Solution cannot be saved as no solution is found.')
