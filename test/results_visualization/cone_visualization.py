#!/bin/env python3
import os
import sys
from os.path import dirname, join, realpath

import numpy as np
import nibabel as nb
import scipy.ndimage as ndi

from nilearn import plotting
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)

from lib.masks import solid_cone, sphere
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid


if __name__ == '__main__':
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    ns = 30

    tk = 25
    overlap = 0
    max_radius = 100

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
        sc = solid_cone(radius=(r_min, r_max), center=centroid, m=1)
        spheres[np.where(sc)] = i + 1

    for i, (r_min, r_max) in enumerate(scales):
        print(f'Scale: {i}')
        for j, angle in enumerate(range(0, 180, ns)):
            print(f'\t- Theta = {angle} deg')
            sph_mask = sphere(center=centroid, radius=(r_min, r_max))
            sph_roi = mni_aseg.get_data() * sph_mask

            shift = (np.array([128, 128, 128])) - np.array(centroid)
            print(f'Center of MNI: {mni_aseg.shape}, centroid {centroid}, difference: {shift}')

            solid_ang_mask = spheres
            solid_ang_mask = ndi.shift(solid_ang_mask, shift=tuple(shift), order=0)

            # Rotate over the angles
            solid_ang_mask = ndi.rotate(solid_ang_mask, angle, axes=(1, 2), reshape=False, order=0)
            solid_ang_mask = ndi.rotate(solid_ang_mask, angle, axes=(0, 2), reshape=False, order=0)

            solid_ang_mask = ndi.shift(solid_ang_mask, shift=tuple(-shift), order=0)
            roi = (solid_ang_mask == i + 1).astype(np.int8)

            # Generate NIFTI images from mask
            nii_roi = nb.Nifti1Image(roi, mni_aseg.affine)
            nii_sph = nb.Nifti1Image(sph_roi, mni_aseg.affine)

            display = plotting.plot_anat(mni_aseg, alpha=0.6) #, cut_coords=centroid)
            display.add_overlay(nii_sph, cmap='Purples', alpha=0.5)
            display.add_overlay(nii_roi, cmap='Reds')
            # plt.show()
            plt.savefig(f'/tmp/rois/scale_{i}_{j}.jpg')

