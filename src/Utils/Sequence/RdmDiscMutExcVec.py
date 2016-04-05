import numpy as np
from scipy.stats import rv_discrete
import copy

__author__ = 'frank.ma'


class RdmDiscMutExcVec(object):
    def __init__(self, density: dict):
        self.density = density

    def draw(self, size: int=5):
        if size > self.density.__len__():
            raise ValueError('drawing size %i is larger than the population size %i' % (size, self.density.__len__()))
        sequence = []
        self.draw_seq(self.density, size=size, seq=sequence)
        return sequence

    @staticmethod
    def draw_seq(density_input, size=5, seq=None):
        if seq is None:
            seq = []
        density = copy.deepcopy(density_input)

        if seq.__len__() >= size:
            seq = seq.sort()
            return

        numbers = np.array(list(density.keys()))
        frequencies = np.array(list(density.values()))
        if abs(sum(frequencies) - 1.0) >= 1e-12:
            raise ValueError('density summation differs from one')

        rand_gen = rv_discrete(values=(numbers, frequencies))
        draw = rand_gen.rvs()
        seq += [draw]
        del density[draw]
        # normalize density
        norm_fact = sum(list(density.values()))
        for key in density:
            density[key] /= norm_fact
        RdmDiscMutExcVec.draw_seq(density, size, seq)
        pass
