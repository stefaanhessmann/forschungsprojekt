import pandas as pd
import numpy as np
import os
import pickle


def generate_input_file(in_path, out_path, r_c=np.inf):
    all_files = os.listdir(in_path)
    number_files = len(all_files)
    output = []
    for i, file in enumerate(all_files[:100]):
        print(i/number_files)
        path = in_path + file
        output.append(transform_input(path, r_c))
    with open(out_path, 'wb') as fp:
        pickle.dump(output, fp)
        fp.close()


def transform_input(path, Rc):
    df_input = pd.read_csv(path, names=['atom', 'x', 'y', 'z', 'unknown'], delimiter='\t', skiprows=2,
                             nrows=19)
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
        r_in_Rc = [df_input.r[np.where(dist_i <= Rc)[0][0]]]
        transformed = [np.dot(trans_matr, r_i) for r_i in r_in_Rc]
        out_data.append(transformed)
    return out_data


def transformation_matrix(old_axes, new_axes):
    transformation_matr = np.linalg.solve(new_axes, old_axes)
    return transformation_matr


def get_new_axes(zero, one, two):
    zero_one = (one - zero) / np.linalg.norm(one - zero)
    zero_two = (two - zero) / np.linalg.norm(two - zero)
    x_axis = zero_one
    z_axis = np.cross(zero_one, zero_two)
    y_axis = np.cross(x_axis, z_axis)
    new_axes = np.vstack((x_axis, y_axis, z_axis))
    return new_axes


def cart_to_sphere(vector):
    is_numpy = False
    if type(vector).__module__ != 'numpy':
        is_numpy = True
    x, y, z = vector
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    sph_vector = [r, theta, phi]
    if is_numpy:
        sph_vector = np.array(sph_vector)
    return sph_vector


if __name__ == '__main__':
    path_to_files = "./../Dataset/dsC7O2H10nsd.xyz/"
    network_input_file = "./../Dataset/NN_input.txt"
    generate_input_file(path_to_files, network_input_file)