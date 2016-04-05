import logging

import numpy as np

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class PrimeSequence(object):
    @staticmethod
    def generate_lp(n: int):
        # generate prime sequence through nested loops
        seq = np.array(range(n + 1))
        flg = np.full(seq.shape, True, dtype=bool)
        flg[[0, 1]] = False
        for idx in seq:
            if not flg[idx]:
                # in the case of none-prime base, just bypassing it
                continue
            else:
                for jdx in range(idx + 1, n + 1):
                    if not flg[jdx]:
                        continue
                    else:
                        if jdx % idx == 0:
                            flg[jdx] = False
        return seq[flg]

    @staticmethod
    def generate_vec(n: int):
        # generate prime sequence through vectorization
        seq = np.array(range(n + 1))
        flg = np.full(seq.shape, True, dtype=bool)
        flg[[0, 1]] = False
        for idx in seq:
            if not flg[idx]:
                # in the case of none-prime base, just bypassing it
                continue
            else:
                flg[idx + 1:][flg[idx + 1:]] = np.logical_and(flg[idx + 1:][flg[idx + 1:]],
                                                              np.mod(seq[idx + 1:][flg[idx + 1:]], idx) != 0)
        return seq[flg]
