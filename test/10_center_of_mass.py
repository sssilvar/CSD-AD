import os

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass

from lib.visualization import show_mri

# Define root
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


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
    # Load subject segmentation file
    aseg_file = os.path.join(root, 'test', 'test_data', 'fsaverage.mgz')
    aseg = nb.load(aseg_file)
    data = aseg.get_data()

    # Consider all the brain volume and exclude non brain regions
    mask = data > 0
    print(mask.shape)

    # Get center of mass (centroid)
    a = get_centroid(mask)
    print('Centroid', a)
    print('Data shape {}'.format(a.shape))
    a = get_centroid(data)
    print('Centroid', a)
    print('Data shape {}'.format(a.shape))

    # Plot the result
    show_mri(mask)
    plt.show()
