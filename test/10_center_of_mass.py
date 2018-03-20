import os

import numpy as np
from scipy.ndimage.measurements import center_of_mass

# Define root
root = os.path.dirname(os.getcwd())


def get_centroid(mat):
    """
    Gets the center of mass coordinates of a n-dimensional matrix
    :param mat: nd-array
    :return: tuple, or list of tuples.
        Coordinates of centers-of-mass.
    """
    centroid_coordinates = np.ceil(center_of_mass(mat)).astype(np.int)
    return centroid_coordinates


if __name__ == '__main__':
    x = np.ones([256, 256, 256])

    print(a)
