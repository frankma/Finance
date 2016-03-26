import logging
import time as tm

import numpy as np

from src.MiniProjects.Maze.maze import Maze
from src.MiniProjects.Maze.solver import MazeSolver

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class MazeSolverBFS(MazeSolver):
    def __init__(self, maze: Maze):
        super().__init__(maze)
        # create a visit log and initialize start point as True
        self.maze_visit_log = np.zeros(maze.maze.shape, dtype=bool)
        self.maze_visit_log[maze.start] = True

    def solve(self):
        self.forward_sweep()
        self.backward_sweep()

    def search_neighbour_nodes(self, current_node: tuple):
        next_nodes = [map(sum, zip(current_node, x)) for x in self.MOVES.values()]
        child_nodes = []
        for next_node in next_nodes:
            nn = tuple(next_node)
            # criteria: 1. reachable; 2. un-visited
            if self.maze.maze[nn] == 0 and not self.maze_visit_log[nn]:
                child_nodes.append(nn)
                self.maze_parent_index[nn] = tuple(current_node)
            # mark this nodes as visited
            self.maze_visit_log[nn] = True
        return child_nodes

    def search_next_state(self, current_state: list):
        next_state = []
        for current_node in current_state:
            next_state = next_state + self.search_neighbour_nodes(current_node)
        return next_state

    def check_termination(self, state: list):
        termination = False
        if state.__len__() == 0:
            termination = True
        else:
            for coordinate in state:
                if coordinate == self.maze.end:
                    termination = True
                    self.found_solution = True
        return termination

    def forward_sweep(self):
        tic = tm.clock()
        moves = 0
        current_state = [self.maze.start]
        termination = self.check_termination(current_state)
        while not termination:
            current_state = self.search_next_state(current_state)
            termination = self.check_termination(current_state)
            moves += 1
        if self.found_solution:
            logger.info('Found solution after %i steps of searching.' % moves)
        else:
            logger.info('Could not find solution after %i steps of searching.' % moves)

        logger.info('Finished forward sweeping, took %r sec.' % round(tm.clock() - tic, 6))

    def backward_sweep(self):
        tic = tm.clock()
        self.maze_path[self.maze.end] = True
        parent = self.maze.end
        while parent != self.maze.start and parent != tuple(self.maze_parent_index[parent]):
            self.maze_path[parent] = True
            parent = tuple(self.maze_parent_index[parent])
        # mark start point as true as last step
        self.maze_path[parent] = True
        logger.info('Finished backward sweeping, took %r sec.' % round(tm.clock() - tic, 6))
