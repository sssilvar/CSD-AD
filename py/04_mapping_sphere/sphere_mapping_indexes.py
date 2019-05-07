#!/bin/env python3
import os
import sys
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb
from scipy import ndimage as ndi

from nilearn import plotting
import matplotlib.pyplot as plt

# Set root folder and append it to path
root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)

from lib.masks import solid_cone
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid

if __name__ == '__main__':
    # Set parameters
    data_folder = '/run/media/ssilvari/HDD/ADNI_FS_registered_flirt'
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    tk = 25
    overlap = 4
    max_radius = 50

    ns = 10  # TODO: Check if it's necessary to change it (Scaling factor

    # Calculate the inner and outer radius
    # for all the spheres: scales
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Number of scales: {} | Scales: {}'.format(len(scales), scales))

    # TODO: Finish the thing
