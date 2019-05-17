import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import affine_transform


def rotate_vol(vol, angles=(0, 0, 0)):
    """
    Rotates a 256x256x256 volume on x, y and z axes.
    :param vol: A 256x256x256 volume
    :param angles: Tuple (angle_x, angle_y, angle_z) in degrees
    :return: Volume rotated (without reshaping)
    """
    # Define angles in radians
    ax, ay, az = (np.deg2rad(angles[0]),
                  0,  # np.deg2rad(angles[1]),
                  np.deg2rad(angles[2]))
    # Create a transformation matrix
    sin_x = np.sin(ax)
    cos_x = np.cos(ax)

    m_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])

    # THERE IS NO ROTATION ON Y-AXIS!!
    # sin_y = np.sin(ay)
    # cos_y = np.cos(ay)
    #
    # m_y = np.array([
    #     [cos_y, 0, sin_y],
    #     [0, 1, 0],
    #     [-sin_y, 0, cos_y]
    # ])
    # END ROTATION ON Y-AXIS

    sin_z = np.sin(az)
    cos_z = np.cos(az)

    m_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    mat = np.dot(m_x, m_z)

    # Add offset
    sx, sy, sz = vol.shape
    sx, sy, sz = (int(sx / 2), int(sy / 2), int(sz / 2))

    offset = np.array([sx, sy, sz], dtype=np.float32)
    offset = np.dot(mat, offset)
    tmp = np.array([sx, sy, sz], dtype=np.float32)
    offset = tmp - offset

    # Apply affine transform
    out = np.zeros_like(vol)
    affine_transform(vol, mat, offset=offset, output=out, order=0)

    # Return value
    return out


def rotate_ndi(vol, centroid, angle=(0, 0)):
    """
    Rotate volume using scipy.ndimage library
    :param vol: Numpy volume
    :param centroid: Anchor point to be used during the rotation
    :param angle: tuple(theta, phi) angles in degrees to rotate
    :return: A rotated volume with same shape as input
    """
    center = np.array(vol.shape) // 2
    shift = center - centroid
    print(f'Center: {center}, centroid: {centroid}, shifting: {shift}')

    shifted_vol = ndi.shift(vol, shift=shift, order=0)
    rotated_vol_theta = ndi.rotate(shifted_vol, axes=(0, 2), angle=angle[0], reshape=False, order=0)
    rotated_vol_phi = ndi.rotate(rotated_vol_theta, axes=(1, 2), angle=angle[1], reshape=False, order=0)
    unshift_vol = ndi.shift(rotated_vol_phi, shift=-shift, order=0)

    return unshift_vol
