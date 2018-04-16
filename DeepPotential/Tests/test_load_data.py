import DeepPotential.Code.load_data as ld
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal


def test_transformation_matr():
    old = np.array([[1, 0, 2], [3, 1, 0], [2, 1, 1]])
    new = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    result = np.array([[3/2, 1, 1], [1/2, -1, 0], [-1/2, 2, 1]])
    assert np.array_equal(result, ld.transformation_matrix(old.T, new.T))


def test_cart_to_sphere():
    vector = np.array([1., 1., -1.])
    theory = np.array([1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(2), 1/np.sqrt(2)])
    result = ld.cart_to_sphere(vector)
    assert_array_almost_equal(theory, result)


def test_transform_vector():
    new_origin = np.array([0, 0, 0])
    vector = np.array([2, -1, 3])
    matrix = np.array([[3/2, 1, 1], [1/2, -1, 0], [-1/2, 2, 1]])
    theory = np.array([5, 2, 0])
    result = ld.transform_vector(vector, new_origin, matrix)
    assert_array_almost_equal(theory, result)


def test_transform_input():
    df_input = pd.DataFrame()
    df_input['atom'] = ['C', 'C', 'C', 'C']
    df_input['x'] = [0., 0., 10., 1.]
    df_input['y'] = [0., 0., 0., 1.]
    df_input['z'] = [1., 0., 0., 1.]
    df_input['unknown'] = [1., 1., 1., 1.]
    theory = np.array([[np.sqrt(2), 0., 1/np.sqrt(2), -1/np.sqrt(2)],
                       [1/np.sqrt(100.5), -10./np.sqrt(100.5), 1/np.sqrt(2), 1/np.sqrt(2)],
                       [np.sqrt(2/3), np.sqrt(2/3), 1/np.sqrt(2), 1/np.sqrt(2)]])
    result = ld.transform_input(df_input, Rc=1000000)[0]
    assert_array_almost_equal(theory, result)


if __name__ == "__main__":
    test_transform_input()