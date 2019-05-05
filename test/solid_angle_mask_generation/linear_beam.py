#!/bin/env python3
from time import time

import numpy as np
from scipy.ndimage import center_of_mass, filters

import nibabel as nb
from nilearn import plotting
import matplotlib.pyplot as plt


def ray_trace(rmin, rmax, theta, phi, center=(128, 128, 128), shape=(256, 256, 256)):
    # Parse shape and center
    sx, sy, sz = shape
    cx, cy, cz = center

    # Define grid and re-center it
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]
    x, y, z = (x - cx, y - cy, z - cz)

    # Create an roi
    theta, phi = np.deg2rad(theta), np.deg2rad(phi)
    th = 0.04

    # Create roi
    sa = np.arctan2(y, x) + 0 * z
    sb = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

    roi_theta = np.logical_and(sa <= theta + th, sa >= theta - th)
    roi_phi = np.logical_and(sb <= phi + th, sb >= phi - th)
    roi = np.logical_and(roi_theta, roi_phi)
    return roi


def hollow_sphere(rmin, rmax, center=(128, 128, 128), shape=(256, 256, 256)):
    # Parse shape and center
    sx, sy, sz = shape
    cx, cy, cz = center

    # Define grid and re-center it
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]
    x, y, z = (x - cx, y - cy, z - cz)

    r = x ** 2 + y ** 2 + z ** 2
    roi = np.logical_and(r >= rmin ** 2, r < rmax ** 2)
    return  roi


if __name__ == '__main__':
    # Set some params
    n_spheres = 4
    tk = 25
    overlap = 5
    step = 6  # Degrees

    scales = [(i * (tk - overlap), ((i + 1) * tk) - (i * overlap)) for i in range(n_spheres)]
    print('Scales: {}'.format(scales))

    # Load image
    nii = nb.load('/home/ssilvari/Documents/temp/MIRIAD/MIRIAD_registered/miriad_188_AD_M_01_MR_1/001_reg.nii.gz')
    vol = nii.get_data()

    # Calculate the center of mass
    com = center_of_mass(vol)
    com = tuple(int(i) for i in list(com))
    print('Center of mass: {}'.format(com))
    vol = filters.sobel(vol)

    # define scales
    for rmin, rmax in scales:
        img = np.zeros([360 // step, 180 // step])
        sphere = hollow_sphere(rmin=rmin, rmax=rmax, center=com)
        print('Image shape: {}'.format(img.shape))

        for i, theta in enumerate(range(0, 180, step)):
            for j, phi in enumerate(range(0, 90, step)):
                roi = np.multiply(
                    ray_trace(rmin=rmin, rmax=rmax, theta=theta, phi=phi, center=com),
                    sphere)
                img[i, j] = np.nan_to_num(np.mean(vol[np.where(roi)]))
                print('Scale: {} | Angle: {}| Mean: {}'.format((rmin, rmax), (theta, phi), img[i, j]))
        plt.imsave('/tmp/{}_to_{}.png'.format(rmin, rmax), img, cmap='gray')
