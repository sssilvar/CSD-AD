import numpy as np
from scipy.ndimage.measurements import center_of_mass

from .masks import sphere, solid_cone
from .transformations import rotate_vol


def get_centroid(mat):
    """
    Gets the center of mass coordinates of a n-dimensional matrix
    :param mat: nd-array
    :return: tuple, or list of tuples.
        Coordinates of centers-of-mass.
    """
    centroid_coordinates = np.ceil(center_of_mass(mat)).astype(np.int)
    return centroid_coordinates


def extract_sub_volume(vol, radius, centroid):
    cx, cy, cz = centroid
    cx_min, cx_max = (cx - radius[1], cx + radius[1])
    cy_min, cy_max = (cy - radius[1], cy + radius[1])
    cz_min, cz_max = (cz - radius[1], cz + radius[1])

    vol_sub = vol[cx_min:cx_max, cy_min:cy_max, cz_min:cz_max]
    center = (int((cx_max - cx_min) / 2), int((cy_max - cy_min) / 2), int((cz_max - cz_min) / 2))

    return vol_sub, center
