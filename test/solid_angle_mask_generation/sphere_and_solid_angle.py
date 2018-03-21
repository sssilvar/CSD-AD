import os

import numpy as np
import time
from numpy import pi
import nibabel as nb
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import affine_transform

import matplotlib.pyplot as plt

from lib.masks import solid_cone
from lib.visualization import show_mri

# Define the root folder
root = os.path.dirname(os.path.dirname(os.getcwd()))

if __name__ == '__main__':
    a = solid_cone(radius=(33, 66))
    print(a.shape)

    cos_gamma = np.cos(pi / 8)
    sin_gamma = np.sin(pi / 8)
    rot_affine = np.array([[1, 0, 0, 0],
                           [0, cos_gamma, -sin_gamma, 0],
                           [0, sin_gamma, cos_gamma, 0],
                           [0, 0, 0, 1]])

    rot_affine_3 = np.array([[cos_gamma, 0, sin_gamma, 0],
                             [0, 1, 0, 0],
                             [-sin_gamma, 0, cos_gamma, 0],
                             [0, 0, 0, 1]])

    rot_affine_0 = np.array([[cos_gamma, 0, sin_gamma],
                             [0, 1, 0],
                             [-sin_gamma, 0, cos_gamma]])

    rot_affine_2 = np.array([[1, 0, 0, 0],
                             [0, cos_gamma, -sin_gamma, 0],
                             [0, sin_gamma, cos_gamma, 0],
                             [0, 0, 0, 1]])

    a_rot = affine_transform(a, rot_affine_0, offset=(-128 * cos_gamma, 0, -128 * sin_gamma), order=1, prefilter=False)

    show_mri(a, slice_xyz=(160, 128, 128))
    show_mri(a_rot, slice_xyz=(160, 128, 128))
    plt.show()
