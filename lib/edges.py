import numpy as np

import nibabel as nb
from scipy.ndimage import sobel


def sobel_magnitude(nii_file=None, nii=None):
    """
    Returns the sobel magnitude space for a given file or volume
    :param nii_file: <str> file path for volume
    :param nii: <nibabel.Nifti1Image> volume already loaded
    :return:
    """
    if nii_file is None and nii is None:
        raise IOError('NIFTI file path or variable not found')
    if nii_file:
        vol = nb.load(nii_file).get_data().astype(np.float)
    if nii:
        vol = nii.get_data().astype(np.float)
    sobel_mode = 'reflect'
    sobel_x = sobel(vol, axis=0, mode=sobel_mode)
    sobel_y = sobel(vol, axis=1, mode=sobel_mode)
    sobel_z = sobel(vol, axis=2, mode=sobel_mode)
    return np.sqrt(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2)
