#!/bin/env python3
import os
import sys
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd
import nibabel as nb
from scipy import ndimage as ndi

from sqlalchemy import create_engine

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
    print(10 * '=' + ' Index saver ' + 10 * '=')
    data_folder = join(os.getenv('HOME'), 'Downloads')
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    tk = 25
    overlap = 9
    max_radius = 100

    ns = 6  # TODO: Check if it's necessary to change it (Scaling factor
    engine = create_engine('sqlite:////dev/shm/indexes_tk_{}_overlap_{}_ns_{}.sqlite'.format(tk, overlap, ns))

    # Print some info
    print('\t- Output folder: %s' % data_folder)
    print('\t- Output file: %s' % '/dev/shm/')

    # Calculate the inner and outer radius
    # for all the spheres: scales
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Number of scales: {} | Scales: {}'.format(len(scales), scales))

    # Get centroid of MNI152
    mni_aseg = nb.load(mni_file)
    centroid = tuple(get_centroid(mni_aseg.get_data() > 0))
    print('Centroid of MNI152: {}'.format(centroid))

    spheres = np.zeros(mni_aseg.shape)
    for i, (r_min, r_max) in enumerate(scales):
        print('Creating sphere: {} ...'.format(i + 1))
        sc = solid_cone(radius=(r_min, r_max), center=centroid)
        spheres[np.where(sc)] = i + 1

    # ==== INDEX CALCULATION ====
    for i, z_angle in enumerate(range(-180, 180, ns)):
        for j, x_angle in enumerate(range(0, 180, ns)):
            print('Processing angles: ({}, {})'.format(z_angle, x_angle))
            solid_ang_mask = rotate_vol(spheres, angles=(x_angle, 0, z_angle))  # ROI
            for k, (r_min, r_max) in enumerate(scales):
                scale = '{}_{}'.format(r_min, r_max)
                ix = np.where(solid_ang_mask == k + 1)
                data = {
                    'scale': scale,
                    'theta': z_angle,
                    'phi': x_angle,
                    'ix': ix[0].tostring(),
                    'iy': ix[1].tostring(),
                    'iz': ix[2].tostring(),
                }
                s = pd.DataFrame()
                s = s.append(pd.Series(data), ignore_index=True)
                s.to_sql('indexes', con=engine, if_exists='append')
                # print(s)
            # Plot for (180, 0) degrees
            if z_angle == -180 and x_angle == 0:
                nii_a = nb.Nifti1Image(solid_ang_mask, mni_aseg.affine)
                nb.save(nii_a, '/tmp/cones.nii')

                display = plotting.plot_anat(mni_aseg)
                display.add_overlay(nii_a)
                plt.show()
    print('DONE!')
