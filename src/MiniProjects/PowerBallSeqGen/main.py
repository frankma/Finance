import numpy as np
from src.MiniProjects.PowerBallSeqGen.SequenceGenerator import SequenceGenerator

__author__ = 'frank.ma'


n = 69
numbers = list(np.linspace(1, n, num=n, dtype=int))
frequencies = list(np.full(n, 1.0 / float(n)))
density = dict(zip(numbers, frequencies))

sg = SequenceGenerator(density)

n_sets = 20
for _ in range(n_sets):
    seq = sg.draw(size=5)
    pb = np.random.randint(1, 27, size=1)
    print(seq, pb)
s