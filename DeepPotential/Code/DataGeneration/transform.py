import numpy as np
from scipy.spatial.distance import squareform, cdist
from Code.DataGeneration.printer import ProgressTimer

def get_spherical(positions):
    """
    Transform 3D cartesian coordinates to spherical coordinates that can
    be used for the nural network input.
    
    Parameters
    ----------
    positions : ndarray
        Array of shape (N, 3) where N is the number of coordinates that
        needs to be transformed.
    
    Returns
    -------
    ndarray
        Array of shape (N, 4) where N is the number of transformed coordinates.
        Transformed coordinates for one position: (1/r, cos(theta), cos(phi), sin(phi)).
        
    """            
    positions = positions.astype(float)
    r = np.linalg.norm(positions, axis=1)
    theta = np.arccos(positions[:, 2]/r)
    phi = np.arctan2(positions[:, 1], positions[:, 0])
    return np.array([1/r, np.cos(theta), np.cos(phi), np.sin(phi)]).T

def change_base(positions, x, y, z, o):
    """
    Calculate the base transformation from the standard basis to the new axes x, y, z.
    
    Parameters
    ----------
    positions : np.array
        3D atom position in the standard basis
    x : np.array
        new x-axis
    y : np.array
        new y-axis
    z : np.array
        new z-axis    
    o : np.array
        new origin
    
    Returns
    -------
    new_positions : np.array
        3D atom position in the new basis.
        Same shape as positions.

    """
    positions -= o
    basis = np.vstack((x, y, z)).T
    basis_inv = np.linalg.inv(basis)
    new_positions = basis_inv.dot(positions.T).T
    return new_positions

def get_input_data(raw_matrix):
    """
    Calculate the training input for the sub-networks from a given molecular configuration.

    Parameters
    ----------
    raw_matrix : np.array
        Matrix of the raw input data for all files and all atoms
    raw_matrix_cols : list
        Column names for the raw_matrix

    Returns
    -------
    X : np.array
        Training data with 'atomtype' and 'relative position-vector' for all other atoms.
    Y : np.array
        Training labels (Mullikan Charge)

    """
    timer = ProgressTimer(len(raw_matrix), print_every=100)
    n_atoms = raw_matrix[0].shape[0]
    h_atoms = np.sum(raw_matrix[0][:, 0] == 1)
    not_H_atoms = n_atoms - h_atoms
    # make a copy
    network_inputs = []
    # create a column for the pos vector
    # loop over all configurations
    for molecule in raw_matrix:
        molecule = molecule[molecule[:, 0].argsort()]
        timer.how_long()
        mol_input = []
        for atom in range(len(molecule)):
            others = np.delete(molecule, atom, axis=0)
            focus_atom = molecule[atom]
            # get distances from focus atom to other atoms
            distances = cdist(focus_atom[1:].reshape(1, 3), molecule[:, 1:])[0]
            zero = focus_atom[1:].astype(float)
            # get nearest atoms that are not H
            nearest = distances.argsort()
            if not_H_atoms >= 3:
                one_id, two_id = nearest[nearest >= h_atoms][1:3]
            else:
                one_id, two_id = nearest[1:3]
            one = molecule[one_id, 1:].astype(float)
            two = molecule[two_id, 1:].astype(float)
            # get new basis vectors
            new_x = one - zero
            new_z = np.cross(new_x, two - zero)
            new_y = np.cross(new_x, new_z)
            # normalize basis vectors
            new_x /= np.linalg.norm(new_x)
            new_y /= np.linalg.norm(new_y)
            new_z /= np.linalg.norm(new_z)
            # sort by distance to origin
            cart_coords = others[:, 1:].astype(float)
            trans_coords = change_base(cart_coords, new_x, new_y,
                                       new_z, zero)
            spherical_coords = get_spherical(trans_coords)
            sort_by_dist = spherical_coords[:, 0].argsort()
            spherical_coords = spherical_coords[sort_by_dist]
            labels = others[sort_by_dist][:, 0]
            spherical_coords = spherical_coords[labels.argsort()]
            net_in_coords = spherical_coords.reshape((n_atoms - 1) * 4).tolist()
            mol_input.append(net_in_coords)
        network_inputs.append(mol_input)
    return network_inputs


if __name__ == '__main__':
    molecules = np.load('./test.npy')
    get_input_data(molecules)