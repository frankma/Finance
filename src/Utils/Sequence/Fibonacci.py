import logging

__author__ = 'frank.ma'

logger = logging.getLogger(__name__)


class Fibonacci(object):
    SEED = [0, 1]

    @staticmethod
    def generate_list(num: int) -> list:
        seq = Fibonacci.SEED.copy()
        for idx in range(num - 2):
            seq.append(seq[-2] + seq[-1])
        return seq

    @staticmethod
    def generate_nth(num: int):
        a, b = Fibonacci.SEED
        for idx in range(num - 2):
            a, b = b, a + b
        return b
