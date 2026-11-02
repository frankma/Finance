from unittest import TestCase

import numpy as np

from src.SolverFD.TriDiagonalMatrix import TriDiagonalMatrix

__author__ = 'frank.ma'


class TestTriDiagonalMatrix(TestCase):
    def test_create(self):
        matrix = TriDiagonalMatrix([1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]).get_matrix()
        self.assertIs(type(matrix), np.ndarray, 'a plain array, not the deprecated np.matrix.')
        np.testing.assert_array_equal(matrix, [[3.0, 6.0, 0.0], [1.0, 4.0, 7.0], [0.0, 2.0, 5.0]])
        with self.assertRaises(ValueError):
            TriDiagonalMatrix.create([1.0], [3.0, 4.0, 5.0], [6.0, 7.0])

    def test_repeated_values(self):
        tdm = TriDiagonalMatrix(-0.5, 2.0, -0.5, 5)
        expected = 2.0 * np.eye(5) - 0.5 * (np.eye(5, k=1) + np.eye(5, k=-1))
        np.testing.assert_array_equal(tdm.get_matrix(), expected)
        np.testing.assert_allclose(tdm.get_inverse() @ tdm.get_matrix(), np.eye(5), atol=1e-14)
        with self.assertRaises(ValueError):
            TriDiagonalMatrix(-0.5, 2.0, -0.5, 2)
