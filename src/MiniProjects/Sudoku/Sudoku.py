import logging
import time as tm

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)

# define constants
BLOCK_X = [(0, 3), (0, 3), (0, 3), (3, 6), (3, 6), (3, 6), (6, 9), (6, 9), (6, 9)]
BLOCK_Y = [(0, 3), (3, 6), (6, 9), (0, 3), (3, 6), (6, 9), (0, 3), (3, 6), (6, 9)]
BLOCK_DICT = dict((idx, [(xdx, ydx) for ydx in range(BLOCK_Y[idx][0], BLOCK_Y[idx][1])
                         for xdx in range(BLOCK_X[idx][0], BLOCK_X[idx][1])]) for idx in range(9))
BLOCK_DICT_REV = dict()
for key in BLOCK_DICT.keys():
    BLOCK_DICT_REV.update({value: key for value in BLOCK_DICT[key]})


class Sudoku():
    def __init__(self,
                 file_path: str):
        tic = tm.time()
        # read the sudoku content
        self.sudoku = np.loadtxt(file_path, delimiter=' ')
        logger.info('Finished data loading, took %r sec.' % round(tm.time() - tic, 6))
        self.check_sudoku(self.sudoku)

    def check_sudoku(self, sudoku: np.ndarray):
        if sudoku.shape != (9, 9):
            raise ValueError('Expecting a 9 * 9 shape, but received %s.' % (sudoku.shape,))
        for idx in range(9):
            row = sudoku[idx, :].tolist()
            col = sudoku[:, idx].tolist()
            block_index = np.array(BLOCK_DICT[idx]).T
            block = sudoku[block_index[0], block_index[1]].tolist()
            self.check_unique(row, 'row', idx)
            self.check_unique(col, 'col', idx)
            self.check_unique(block, 'block', idx)

    @staticmethod
    def check_unique(value_list: list,
                     value_type: str,
                     value_idx: int):
        # make a dictionary of inputs, note only check 1 to 9, bypass zeros
        array_dict = dict([(i, value_list.count(i)) for i in range(1, 10)])
        if max(array_dict.values()) > 2:
            raise ValueError('Duplicated values found at %s %i %r' % (value_type, value_idx, value_list))
