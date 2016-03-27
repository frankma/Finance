import logging

from src.MiniProjects.Sudoku.Sudoku import Sudoku
from src.MiniProjects.Sudoku.SudokuSolverEC import SudokuSolverEC

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

problem_number = 3
input_path = './/resource//sudoku_%i.txt' % problem_number
output_path = './/resource//sudoku_%i_solution.txt' % problem_number

sudoku = Sudoku(input_path)
solver = SudokuSolverEC(sudoku)
solver.solve()
solver.print_solution(output_path, print_partial=True)
