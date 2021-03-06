import os

import numpy as np
import time
from numpy import pi
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

# Define the root folder
root = os.path.dirname(os.path.dirname(os.getcwd()))


def sphere(shape=(256, 256, 256), radius=(1, 10), center=(128, 128, 128),
           theta_range=(-pi, pi), phi_range=(-pi / 2, pi / 2)):
    # Create variables for simplicity
    sx, sy, sz = shape
    r_min, r_max = radius
    cx, cy, cz = center
    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    # Define a coordinate system
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]

    # Create an sphere in the range of r, theta and phi
    x = x - cx
    y = y - cy
    z = z - cz

    # For radius range, theta range and phi range
    eqn_mag = x ** 2 + y ** 2 + z ** 2
    eqn_theta = np.arctan2(y, x)
    eqn_theta = np.repeat(eqn_theta[:, :, np.newaxis], sz, axis=2).squeeze()

    eqn_phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)

    # Generate masks
    mask_radius = np.logical_and(eqn_mag > r_min ** 2, eqn_mag <= r_max ** 2)
    mask_theta = np.logical_and(eqn_theta >= theta_min, eqn_theta <= theta_max)
    mask_phi = np.logical_and(eqn_phi >= phi_min, eqn_phi <= phi_max)

    # Generate a final mask
    mask = np.logical_and(mask_radius, mask_phi)
    mask = np.logical_and(mask, mask_theta)

    return mask


def circle(shape=(256, 256), radius=(1, 10), center=(128, 128), ang_range=(-pi, pi)):
    # Create variables for simplicity
    sx, sy = shape
    r_min, r_max = radius
    cx, cy = center
    a_min, a_max = ang_range

    # Define a coordinate system (cartesian)
    x, y = np.ogrid[0:sx, 0:sy]

    # Create a circle
    eqn_mag = (x - cx) ** 2 + (y - cy) ** 2
    eqn_angle = np.arctan2((y - cy), (x - cx))

    radius_mask = np.logical_and(eqn_mag > r_min ** 2, eqn_mag <= r_max ** 2)
    angle_mask = np.logical_and(eqn_angle >= a_min, eqn_angle <= a_max)

    # assembly the final mask
    mask = np.logical_and(radius_mask, angle_mask)

    return mask


def cone(shape=(256, 256, 256), center=(128, 128, 128), r=100):
    """Draw a cone"""
    # Create variables for simplicity
    sx, sy, sz = shape
    cx, cy, cz = center

    # Define and ordinate system (cartesian)
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]
    x = x - cx
    y = y - cy
    z = z - cz

    # Draw a cone
    eqn_cone = np.sqrt(x ** 2 + y ** 2) - z
    mask = eqn_cone <= -r

    return mask


def solid_cone(radius=(100, 110), center=(128, 128, 128)):
    # Define variables for simplicity
    r_min, r_max = radius
    cx, cy, cz = center

    # Create a Sphere and a cone
    sphere_vol = sphere(radius=(r_min, r_max), center=center)
    cone_vol = cone(r=r_min)
    mask = sphere_vol * cone_vol

    vol = mask.astype(np.int8)
    vol[cx, cy, cz + r_max] = 2

    return vol


def show_mri(vol, slice_xyz=(128, 128, 128)):
    img_x = vol[:, :, slice_xyz[0]].reshape((256, 256))
    img_y = vol[:, slice_xyz[1], :].reshape((256, 256))
    img_z = vol[slice_xyz[2], :, :].reshape((256, 256))

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_x, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img_y, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img_z, cmap='gray')


if __name__ == '__main__':
    vol = solid_cone(radius=(50, 60))

    # theta_r = pi/2
    # rx = [[1, 0, 0],
    #       [0, np.cos(theta_r), -np.sin(theta_r)],
    #       [0, np.sin(theta_r), np.cos(theta_r)]]
    #
    # vol = vol * np.array([[0,1], [-1, 0]])
    cos_gamma = np.cos(pi / 4)
    sin_gamma = np.sin(pi / 4)
    rot_affine = np.array([[1, 0, 0, 0],
                           [0, cos_gamma, -sin_gamma, 0],
                           [0, sin_gamma, cos_gamma, 0],
                           [0, 0, 0, 1]])

    nii = nb.Nifti1Image(vol, affine=rot_affine)

    file_output = '/home/sssilvar/Documents/tmp/test.mgz'
    nb.save(nii, file_output)

    p_mgz = os.path.join(os.path.dirname(file_output), 'p.mgz')
    os.system('mri_convert %s %s -c' % (file_output, p_mgz))
    mgz = nb.load('/home/sssilvar/Documents/tmp/p.mgz')
    img = mgz.get_data()

    show_mri(vol)
    show_mri(img)
    plt.show()
