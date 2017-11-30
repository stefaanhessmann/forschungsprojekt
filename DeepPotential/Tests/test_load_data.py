from DeepPotential.Code.load_data import transformation_matrix
import numpy as np


def test_transformation_matr():
    old = np.array([[1, 0, 2], [3, 1, 0], [2, 1, 1]])
    new = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result = np.array([[3/2, 1, 1], [1/2, -1, 0], [-1/2, 2, 1]])
    assert np.array_equal(result, transformation_matrix(old.T, new.T))

