def read_xyz(filename):
    """Read filename in XYZ format and return lists of atoms and coordinates.
    If number of coordinates do not agree with the statd number in
    the file it will raise a ValueError.
    """
    coordinates = []
    xyz = open(filename)
    n_atoms = int(xyz.readline())
    title = xyz.readline()
    counter = 2
    for line in xyz:
        if counter < 2:
            counter += 1
        else:
            atom, x, y, z, n = line.split()
            coordinates.append([int(n), float(x), float(y), float(z)])
            if counter == 4:
                counter = 0
            else:
                counter += 1
    xyz.close()
    return coordinates
