import os

import numpy as np
import time
from numpy import pi
import nibabel as nb
import matplotlib.pyplot as plt


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


def sphere_to_plane(vol, shape=(256, 256, 256), radius_range=(), center=(128,128,128), step=1):
    img_2d = np.zeros([180, 360])

    for i, theta in enumerate(range(-180, 180, step)):
        for j, phi in enumerate(range(-90, 90, step)):
            t_min = np.deg2rad(theta)
            t_max = np.deg2rad(theta + step)

            p_min = np.deg2rad(phi)
            p_max = np.deg2rad(phi + 1)

            # Calculate a sphere mask
            sph = sphere(shape=shape, radius=radius_range, center=center,
                         theta_range=(t_min, t_max), phi_range=(p_min, p_max))

            # Mask the volume
            vol_masked = vol * sph

            # extract point
            point_value = vol_masked.sum() / sph.sum()
            img_2d[i, j] = point_value
            print('Point (%d, %d) of (360/180)' % (i, j))

    return img_2d


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
    mgz = nb.load(os.path.join(root, 'test', 'test_data', '941_S_1363.mgz'))
    img = mgz.get_data()

    r_min = 33
    r_max = 66

    img_2d = sphere_to_plane(img, shape=img.shape, radius_range=(r_min, r_max), step=1)

    filename_output = os.path.join(root, 'output', 'sphere_to_map_scale_%d_to_%d.jpg') % (r_min, r_max)
    plt.imsave(filename_output, img_2d, cmap='gray')







