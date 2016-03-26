import logging

import numpy as np

from src.MiniProjects.Maze.Maze import Maze

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class MazeSolver(object):
    MOVES = dict(N=(-1, 0), W=(0, 1), S=(1, 0), E=(0, -1))
    found_solution = False

    def __init__(self,
                 maze: Maze):
        self.maze = maze
        self.maze_path = np.zeros(self.maze.maze.shape, dtype=bool)
        self.maze_parent_index = self.__init_parent_index(maze.maze)

    def solve(self):
        pass

    def print_solution(self,
                       output: str):
        if not self.found_solution:
            logger.info('No solution is found for the given maze.')
        else:
            maze_display = np.chararray(self.maze.maze.shape)
            maze_display.fill(str(' '))
            maze_display[self.maze.maze == 1] = '#'
            maze_display[self.maze_path] = 'X'
            maze_display[self.maze.start] = 'S'
            maze_display[self.maze.end] = 'E'
            np.savetxt(output, maze_display.decode(), fmt='%s', delimiter='')
            logger.info('Maze solution is printed in file %s.' % output)

    @staticmethod
    def __init_parent_index(maze: np.array):
        return np.array([[(x, y) for y in range(maze.shape[1])] for x in range(maze.shape[0])])
