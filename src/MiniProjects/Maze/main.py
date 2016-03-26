import logging

from src.MiniProjects.Maze.Maze import Maze
from src.MiniProjects.Maze.MazeSolverBFS import MazeSolverBFS

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)

problem_number = 3
input_path = './/resource//maze_%i.txt' % problem_number
output_path = './/resource//maze_%i_solution.txt' % problem_number

maze = Maze(input_path)
maze.print_maze()
solver = MazeSolverBFS(maze)
solver.solve()
solver.print_solution(output_path)
