import numpy as np
from numpy import pi


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

    # vol = mask.astype(np.int8)
    # vol[cx, cy, cz + r_max] = 2

    return mask
