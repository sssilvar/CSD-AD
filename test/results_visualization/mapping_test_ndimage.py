#!/bin/env python3
import os
import sys
import shutil
from os.path import dirname, join, realpath, isdir

import numpy as np
import scipy.ndimage as ndi

import nibabel as nb
from nilearn import plotting
import matplotlib.pyplot as plt

# Define root and add it to system path
root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)

from lib.masks import solid_cone, sphere
from lib.geometry import get_centroid
from lib.transformations import rotate_ndi
from lib.edges import sobel_magnitude


# def rotate(vol, centroid, angle=(0, 0)):
#     center = np.array(vol.shape) // 2
#     shift = center - centroid
#     print(f'Center: {center}, centroid: {centroid}, shifting: {shift}')
#
#     shifted_vol = ndi.shift(vol, shift=shift, order=0)
#     rotated_vol_theta = ndi.rotate(shifted_vol, axes=(1, 2), angle=angle[1], reshape=False, order=0)
#     rotated_vol_phi = ndi.rotate(rotated_vol_theta, axes=(0, 2), angle=angle[0], reshape=False, order=0)
#     unshift_vol = ndi.shift(rotated_vol_phi, shift=-shift, order=0)
#
#     return unshift_vol


def generate_solid_cones(scales, m=1):
    cones = np.zeros(mni_aseg.shape)
    for i, (r_min, r_max) in enumerate(scales):
        print('Creating sphere: {} - {} ...'.format((i + 1), (r_min, r_max)))
        sc = solid_cone(radius=(r_min, r_max), center=centroid, m=m)
        cones[np.where(sc)] = i + 1
    return cones


if __name__ == '__main__':
    mni_file = join(root, 'param', 'FSL_MNI152_FreeSurferConformed_1mm.nii')
    out_folder = '/tmp/rois'
    ns = 9

    tk = 25
    overlap = 0
    max_radius = 100

    # Create output folder
    if not isdir(out_folder):
        os.mkdir(out_folder)
    else:
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)

    # Calculate the inner and outer radius
    # for all the spheres: scales
    n_spheres = max_radius // (tk - overlap)
    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Number of scales: {} | Scales: {}'.format(len(scales), scales))

    # Get centroid of MNI152
    mni_aseg = nb.load(mni_file)
    mni_data = sobel_magnitude(nii=mni_aseg)
    centroid = tuple(get_centroid(mni_data > 0))
    print('Centroid of MNI152: {}'.format(centroid))

    # Get cones from scales
    cones = generate_solid_cones(scales, m=8)

    mapped_imgs = [np.zeros([360 // ns, 180 // ns]) for i in scales]

    for theta_i, theta in enumerate(range(-180, 180, ns)):
        for phi_i, phi in enumerate(range(-90, 90, ns)):
            print(f'Rotating ({theta}, {phi}) degrees...')
            roi = rotate_ndi(vol=cones, centroid=centroid, angle=(theta, phi))
            # Create NIFTI and plot them
            # roi_nii = nb.Nifti1Image(roi, mni_aseg.affine)
            # display = plotting.plot_anat(mni_aseg, alpha=0.8, title=f'Theta: {theta} deg')
            # display.add_overlay(roi_nii, cmap='jet')
            # plt.show()
            for i, (r_min, r_max) in enumerate(scales):
                mean_pix = mni_data[np.where(roi == (i + 1))].mean()
                mapped_imgs[i][theta_i, phi_i] = mean_pix
                # print(f'Mean: {mean_pix}')

    # Plot results
    for i, (r_min, r_max) in enumerate(scales):
        plt.imshow(mapped_imgs[i], cmap='gray')
        plt.savefig(join(out_folder, f'{r_min}_to_{r_max}.png'))
