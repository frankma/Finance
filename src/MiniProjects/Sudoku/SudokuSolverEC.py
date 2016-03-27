import logging
import time as tm

import numpy as np

from src.MiniProjects.Sudoku.Sudoku import Sudoku, BLOCK_X, BLOCK_Y, BLOCK_DICT_REV
from src.MiniProjects.Sudoku.SudokuSolver import SudokuSolver

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class SudokuSolverEC(SudokuSolver):
    def __init__(self,
                 problem: Sudoku):
        tic = tm.time()
        super().__init__(problem)
        # first make a copy of inputs
        self.solution = self.problem.sudoku
        # create a complete map of sudoku candidates and a corresponding determination matrix
        self.candidates = [[list(range(1, 10)) for x in range(9)] for y in range(9)]
        self.determined = np.zeros(self.problem.sudoku.shape, dtype=bool)
        self.visited = np.zeros(self.problem.sudoku.shape, dtype=bool)
        # initial elimination of cell value candidates based on inputs
        self.elimination_init()
        logger.info('Finished solver initialization, took %r sec.' % round(tm.time() - tic, 6))

    def solve(self):
        tic = tm.time()
        iteration_count = 1
        determined_n_cur = sum(sum(self.determined))
        determined_n_pre = 0
        while not np.all(self.determined) and determined_n_cur > determined_n_pre:
            determined_n_pre = determined_n_cur
            self.elimination()
            determined_n_cur = sum(sum(self.determined))
            iteration_count += 1

        if determined_n_cur == np.size(self.problem.sudoku):
            self.found_solution = True
            self.solution = np.array(self.candidates)
            logger.info('Solution found after %i iterations.' % iteration_count)
        else:
            # save the last stage determined values
            for xdx in range(self.solution.shape[0]):
                for ydx in range(self.solution.shape[1]):
                    if self.determined[xdx, ydx]:
                        self.solution[xdx, ydx] = self.candidates[xdx][ydx]
            logger.warning('CANNOT find solution after %i iterations.' % iteration_count)

        logger.info('Finished calling the solver, took %r sec.' % round(tm.time() - tic, 6))

    def elimination_init(self):
        for xdx in range(self.problem.sudoku.shape[0]):
            for ydx in range(self.problem.sudoku.shape[1]):
                if not self.problem.sudoku[(xdx, ydx)] == 0:
                    # determined candidates will be de-listed
                    self.candidates[xdx][ydx] = self.problem.sudoku[(xdx, ydx)]
                    self.determined[(xdx, ydx)] = True

    def elimination(self):
        for xdx in range(self.problem.sudoku.shape[0]):
            for ydx in range(self.problem.sudoku.shape[1]):
                if self.determined[(xdx, ydx)] and not self.visited[(xdx, ydx)]:
                    self.screen_candidates(xdx, ydx)
                    self.visited[(xdx, ydx)] = True

    def screen_candidates(self,
                          xdx: int,
                          ydx: int):
        fixed_value = self.candidates[xdx][ydx]
        # check repeated values in given row
        for jdx in range(len(self.candidates[xdx][:])):
            if isinstance(self.candidates[xdx][jdx], list) and fixed_value in self.candidates[xdx][jdx]:
                self.candidates[xdx][jdx].remove(fixed_value)
                self.check_determination(xdx, jdx)
        # check repeated values in given col
        for idx in range(len(self.candidates[:][ydx])):
            if isinstance(self.candidates[idx][ydx], list) and fixed_value in self.candidates[idx][ydx]:
                self.candidates[idx][ydx].remove(fixed_value)
                self.check_determination(idx, ydx)
        # check repeated values in nested block
        block_n = BLOCK_DICT_REV[(xdx, ydx)]
        for idx in range(BLOCK_X[block_n][0], BLOCK_X[block_n][1]):
            for jdx in range(BLOCK_Y[block_n][0], BLOCK_Y[block_n][1]):
                if isinstance(self.candidates[idx][jdx], list) and fixed_value in self.candidates[idx][jdx]:
                    self.candidates[idx][jdx].remove(fixed_value)
                    self.check_determination(idx, jdx)

    def check_determination(self,
                            xdx: int,
                            ydx: int):
        value = self.candidates[xdx][ydx]
        if len(value) == 1:
            self.candidates[xdx][ydx] = value[0]
            self.determined[xdx][ydx] = True
            # logger.debug('row %i, col %i, determined value = %i.' % (xdx, ydx, value[0]))
