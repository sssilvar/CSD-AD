#!/bin/env python3
import os
import sys
from os.path import join, dirname, realpath, basename

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
from lib.transformations import rotate_ndi
from lib.geometry import extract_sub_volume, get_centroid

if __name__ == '__main__':
    # Set parameters
    print(10 * '=' + ' Index saver ' + 10 * '=')
    data_folder = join(os.getenv('HOME'), 'Downloads')
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')

    tk = int(sys.argv[1])
    overlap = int(sys.argv[2])
    max_radius = 100
    ns = 1  # TODO: Check if it's necessary to change it (Scaling factor
    sql_file = f'/dev/shm/indexes_tk_{tk}_overlap_{overlap}_ns_{ns}.sqlite'
    disk_sql_file = join(os.getenv('HOME'), 'Downloads', basename(sql_file))

    # Print some info
    print('\t- Output folder: %s' % data_folder)
    print('\t- Output file: %s' % sql_file)
    print('\t- Output file (in disk): %s' % disk_sql_file)

    # Calculate the inner and outer radius
    # for all the spheres: scales
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Number of scales: {} | Scales: {}'.format(len(scales), scales))

    # Get centroid of MNI152
    mni_aseg = nb.load(mni_file)
    centroid = tuple(get_centroid(mni_aseg.get_data() > 0))
    print('Centroid of MNI152: {}'.format(centroid))

    # Make SQL connection
    engine = create_engine(f'sqlite:///{sql_file}')

    # Initialize spheres
    spheres = np.zeros(mni_aseg.shape)
    for i, (r_min, r_max) in enumerate(scales):
        print('Creating sphere: {} ...'.format(i + 1))
        sc = solid_cone(radius=(r_min, r_max), center=centroid)
        spheres[np.where(sc)] = i + 1

    # ==== INDEX CALCULATION ====
    for i, theta in enumerate(range(-180, 180, ns)):
        for j, phi in enumerate(range(-90, 90, ns)):
            print('Processing angles: ({}, {})'.format(theta, phi))
            solid_ang_mask = rotate_ndi(spheres, centroid=centroid, angle=(theta, phi))  # ROI
            for k, (r_min, r_max) in enumerate(scales):
                scale = '{}_{}'.format(r_min, r_max)
                ix = np.where(solid_ang_mask == k + 1)
                data = {
                    'scale': scale,
                    'theta': theta,
                    'phi': phi,
                    'ix': ix[0].tostring(),
                    'iy': ix[1].tostring(),
                    'iz': ix[2].tostring(),
                }
                s = pd.DataFrame()
                s = s.append(pd.Series(data), ignore_index=True)
                s.to_sql('indexes', con=engine, if_exists='append')
                # print(s)
            # Plot for (180, 0) degrees
            if theta == 0 and phi == 0:
                nii_a = nb.Nifti1Image(solid_ang_mask, mni_aseg.affine)
                nb.save(nii_a, '/tmp/cones.nii')

                display = plotting.plot_anat(mni_aseg)
                display.add_overlay(nii_a)
                plt.show()
    print('DONE!')
    os.system(f'mv {sql_file}  {disk_sql_file}')
