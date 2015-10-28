import numpy as np

__author__ = 'frank.ma'


class TriDiagonalMatrix(object):
    def __init__(self, lower: np.array or float, center: np.array or float, upper: np.array or float, n: int = None):
        if n is None:
            self.lower = lower
            self.center = center
            self.upper = upper
        elif n > 2:
            self.lower = np.full(n - 1, lower)
            self.center = np.full(n, center)
            self.upper = np.full(n - 1, upper)
        self.matrix = self.create(self.lower, self.center, self.upper)

    def get_matrix(self):
        return self.matrix

    def get_inverse(self):
        return np.linalg.inv(self.matrix)

    @staticmethod
    def create(lower: np.array or list, center: np.array or list, upper: np.array or list) -> np.matrix:
        n = center.__len__()
        if lower.__len__() != upper.__len__() != n - 1:
            raise ValueError('invalid length of lower, center and upper vector, expect (n - 1, n, n - 1) shape')
        matrix = np.zeros((n, n), dtype=float)
        # header set
        matrix[0][0] = center[0]
        matrix[0][1] = upper[0]
        # regular set
        for idx in range(1, n - 1):
            matrix[idx][idx - 1] = lower[idx - 1]
            matrix[idx][idx] = center[idx]
            matrix[idx][idx + 1] = upper[idx]
        # footer set
        matrix[n - 1][n - 2] = lower[n - 2]
        matrix[n - 1][n - 1] = center[n - 1]

        return np.matrix(matrix)
