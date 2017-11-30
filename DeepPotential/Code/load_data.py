import pandas as pd
import numpy as np
import os
import pickle


def generate_input_file(in_path, out_path, r_c=100000000):
    all_files = os.listdir(in_path)
    number_files = len(all_files)
    output = []
    for i, file in enumerate(all_files[:1000]):
        total = 30
        progress = int(i/number_files*total)
        print('[', progress*'#', (total-progress)*' ', ']')
        path = in_path + file
        df_input = pd.read_csv(path, names=['atom', 'x', 'y', 'z', 'unknown'], delimiter='\t', skiprows=2,
                               nrows=19)
        output.append(transform_input(df_input, r_c))
    with open(out_path, 'wb') as fp:
        pickle.dump(output, fp)
        fp.close()


def transform_input(df_input, Rc):
    old_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    df_input['r'] = df_input[['x', 'y', 'z']].values.tolist()
    # df_input.drop(['x', 'y', 'z'], axis=1)
    system_size = len(df_input)
    dist_matr = np.zeros((system_size, system_size))
    out_data = []
    for i in range(system_size):
        # neighbours for particle 1
        for j in range(i, system_size):
            if i == j:
                dist_ij = np.inf
            else:
                dist_ij = np.linalg.norm(np.array(df_input.r[i]) - np.array(df_input.r[j]))
            dist_matr[i, j] = dist_ij
            dist_matr[j, i] = dist_ij
        dist_i = dist_matr[i, :]
        dist_i_clean = [r for j, r in enumerate(dist_i) if df_input.atom[j] != 'H']
        dist_i_clean = sorted(dist_i_clean)

        one, two = np.array(df_input.r[np.where(dist_i == dist_i_clean[0])[0][0]]), \
                   np.array(df_input.r[np.where(dist_i == dist_i_clean[1])[0][0]])
        zero = np.array(df_input.r[i])

        new_axes = get_new_axes(zero, one, two)
        trans_matr = transformation_matrix(old_axes, new_axes)
        r_in_Rc = df_input.r[np.where(dist_i <= Rc)[0]].as_matrix()

        transformed = [transform_vector(r_i, zero, trans_matr) for r_i in r_in_Rc]
        transformed = [cart_to_sphere(r_i) for r_i in transformed]
        out_data.append(transformed)
    return out_data


def transformation_matrix(old_axes, new_axes):
    transformation_matr = np.linalg.solve(new_axes, old_axes)
    return transformation_matr


def transform_vector(vector, new_origin, trans_matr):
    # translate coordinate system:
    vector -= new_origin
    vector = np.dot(trans_matr, vector)
    return vector


def get_new_axes(zero, one, two):
    """
    Parameters
    ----------
    zero: array
        new origin
    one: array
        nearest neighbour
    two: array
        second nearest neighbour

    Return
    ------
    np.ndarray with shape (3, 3)
        new axes
    """
    zero_one = (one - zero) / np.linalg.norm(one - zero)
    zero_two = (two - zero) / np.linalg.norm(two - zero)
    x_axis = zero_one
    z_axis = np.cross(zero_one, zero_two)
    y_axis = np.cross(x_axis, z_axis)
    new_axes = np.vstack((x_axis, y_axis, z_axis))
    return new_axes


def cart_to_sphere(vector):
    """
    Parameters
    ----------
    vector: array or list
        cartesian coordinates

    Return
    ------
    array or list
        spherical coordinates as [1/r, cos(theta), cos(phi), sin(phi)]

    """
    is_numpy = False
    if type(vector).__module__ != 'numpy':
        is_numpy = True
    x, y, z = vector
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    network_vector = [1/r, np.cos(theta), np.cos(phi), np.sin(phi)]
    if is_numpy:
        network_vector = np.array(network_vector)
    return network_vector


if __name__ == '__main__':
    path_to_files = "./../Dataset/dsC7O2H10nsd.xyz/"
    network_input_file = "./../Dataset/NN_input.txt"
    generate_input_file(path_to_files, network_input_file)
    #old_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #new_axes = np.array([[0, 0, -1], [1, 1, 0], [-1, 1, 0]])
    #trans = transformation_matrix(old_axes, new_axes)
    #print(trans)