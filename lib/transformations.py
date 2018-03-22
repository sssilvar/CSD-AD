import numpy as np
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
                  np.deg2rad(angles[1]),
                  np.deg2rad(angles[2]))
    # Create a transformation matrix
    sin_x = np.sin(ax)
    cos_x = np.cos(ax)

    m_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])

    sin_y = np.sin(ay)
    cos_y = np.cos(ay)

    m_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    sin_z = np.sin(az)
    cos_z = np.cos(az)

    m_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    mat = np.dot(np.dot(m_x, m_y), m_z)
    # Add offset
    offset = np.array([128, 128, 128], dtype=np.float64)
    offset = np.dot(mat, offset)
    tmp = np.array([128, 128, 128], dtype=np.float64)
    offset = tmp - offset

    # Apply affine transform
    out = np.zeros_like(vol)
    affine_transform(vol, mat, offset=offset, output=out, order=0)

    # Return value
    return out
