import logging
import sys
from unittest import TestCase

import numpy as np

from src.Utils.Sequence.RdmDiscMutExcVec import RdmDiscMutExcVec

__author__ = 'frank.ma'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class TestRdmDiscExcVec(TestCase):
    def test_power_ball(self):
        logger.info('%s starts' % sys._getframe().f_code.co_name)
        n = 69
        numbers = list(np.linspace(1, n, num=n, dtype=int))
        frequencies = list(np.full(n, 1.0 / float(n)))
        density = dict(zip(numbers, frequencies))

        sg = RdmDiscMutExcVec(density)

        n_sets = 20
        for _ in range(n_sets):
            seq = sg.draw(size=5)
            pb = np.random.randint(1, 27, size=1)
            logger.debug('%s; %i' % (seq, pb))
        logger.info('%s passes' % sys._getframe().f_code.co_name)
        pass
