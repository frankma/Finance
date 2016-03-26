import logging
import time as tm

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Maze(object):
    def print_maze(self):
        logger.info('size: [%i, %i]' % self.size)
        logger.info('start: [%i, %i]' % self.start)
        logger.info('end: [%i, %i]' % self.end)
        logger.info('maze:')
        logger.info(self.maze)
        return None

    def __init__(self, file_path: str):
        tic = tm.clock()
        # read header
        data = open(file_path)
        self.size = self.__read_line(data.readline())
        self.start = self.__read_line(data.readline())
        self.end = self.__read_line(data.readline())
        data.close()
        # read maze values
        self.maze = np.loadtxt(file_path, delimiter=' ', skiprows=3)
        logger.info('Finished data loading, took %r sec.' % round(tm.clock() - tic, 6))

    @staticmethod
    def __read_line(s: str):
        return tuple([int(x) for x in s.strip().split(sep=' ')])
