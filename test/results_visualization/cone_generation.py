#!/bin/env python3
__description__ = """
This is a try for an optimized and faster generation 
of the solid cone involved in the mapping from a hollow 
sphere to a plane in my project.
"""
__date__ = "Jan 21st 2019"

# Start the magic!
import numpy as np

import nibabel as nb
from nilearn import plotting, image
import matplotlib.pyplot as plt


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def sph2cart(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(phi)

    return x, y, z

def generate_cone_mask_2d(rmin, rmax, theta, alpha):
    if theta < -180 or theta > 180:
        raise Exception("Theta must be beween -180 and 180 degrees.")

    x, y = np.ogrid[-128:128, -128:128]
    eqn_circle = x ** 2 + y ** 2
    hollow_cicle_img = np.logical_and(eqn_circle >= rmin ** 2, eqn_circle <= rmax ** 2)

    sx, sy = pol2cart(rmin, np.deg2rad(theta))

    if -135 <= theta <= 135:
        eqn_cone = np.arctan2(y - sy , x - sx)
        cone_img = np.logical_and(eqn_cone >= np.deg2rad(theta - alpha), eqn_cone <= np.deg2rad(theta + alpha))
    elif theta > 0:
        eqn_cone_a = np.arctan2(y - sy , x - sx)
        eqn_cone_b = - np.arctan2(y - sy , x - sx)
        cone_img_a = np.logical_and(eqn_cone_a >= np.deg2rad(theta - alpha), eqn_cone_a <= np.deg2rad(theta + alpha))
        cone_img_b = np.logical_and(eqn_cone_b >= np.deg2rad(180 - (theta + alpha - 180)), eqn_cone_b <= np.deg2rad(180))
        cone_img = np.logical_or(cone_img_a, cone_img_b)
    elif theta < 0:
        eqn_cone_a = np.arctan2(y - sy , x - sx)
        eqn_cone_b = np.arctan2(y - sy , x - sx)
        cone_img_a = np.logical_and(eqn_cone_a >= np.deg2rad(180 + (theta - alpha + 180)), eqn_cone_a <= np.deg2rad(180))
        cone_img_b = np.logical_and(eqn_cone_b >= np.deg2rad(theta - alpha), eqn_cone_b <= np.deg2rad(theta + alpha))
        cone_img = np.logical_or(cone_img_a, cone_img_b)

    return np.logical_and(hollow_cicle_img, cone_img)



if __name__ == "__main__":
    # Create a simple cone
    rmin, rmax = 25, 75
    theta = 0
    phi = 90
    alpha = 30  # Aperture angle

    # Convert to Radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    alpha = np.deg2rad(alpha)

    # cone_img = generate_cone_mask_2d(rmin, rmax, theta, alpha)

    # # Plot image
    # fig, axes = plt.subplots(ncols=1, nrows=1)
    # axes.imshow(cone_img, cmap='gray')

    # plt.show()

    # Generate 3D cone
    x, y, z = np.ogrid[-128:128, -128:128, -128:128]
    eqn_sphere = x ** 2 + y ** 2 + z **2
    hollow_sphere_img = np.logical_and(eqn_sphere >= rmin ** 2, eqn_sphere <= rmax ** 2)

    # Rays
    cx, cy, cz = sph2cart(rmin, theta, phi)
    x, y, z = (x-cx), (y-cy), (z-cz)
    eqn_theta = np.arctan2(y*z, x*z)
    eqn_phi = np.arctan2( z * y, x * y )

    ray_img_theta = np.logical_and(eqn_theta >= (theta - alpha), eqn_theta <= (theta + alpha))
    ray_img_phi = np.logical_and(eqn_phi >= (phi - alpha), eqn_phi <= (phi + alpha))
    
    ray_img = np.logical_and(ray_img_theta, ray_img_phi)
    roi_img = np.logical_and(ray_img, hollow_sphere_img)

    # print('Initial point: {}'.format((cx, cy, cz)))
    print('Angles: ({},{}) | Aperture: {}'.format(theta, phi, alpha))

    # Create NIFTI images
    nii_sphere = nb.Nifti1Image(hollow_sphere_img.astype(np.int8), np.eye(4))
    nii_ray = nb.Nifti1Image(ray_img.astype(np.float), np.eye(4))
    nii_roi = nb.Nifti1Image(roi_img.astype(np.int8), np.eye(4))

    nb.save(nii_ray, '/tmp/test.nii.gz')

    # Plot
    # cx, cy, cz = 128 + cx, 128 + cy, 128 + cz
    # display = plotting.plot_anat(nii_sphere, alpha=0.2, cut_coords=(cx, cy, cz))
    # display.add_overlay(nii_ray)
    # plt.show()
