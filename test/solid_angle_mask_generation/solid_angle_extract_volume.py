import os
import sys

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(root)
from lib.geometry import solid_cone, sphere
from lib.visualization import show_mri
from lib.geometry import get_centroid, extract_sub_volume

if __name__ == '__main__':
    # Define file names
    vol_filename = os.path.join(root, 'test', 'test_data', '941_S_1363.mgz')
    aseg_filename = os.path.join(root, 'test', 'test_data', 'mri', 'aseg.mgz')

    # Load images
    vol = nb.load(vol_filename).get_data()
    aseg = nb.load(aseg_filename).get_data()

    # Define radius and center
    radius = (40, 100)
    centroid = tuple(get_centroid(aseg > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    # Create a binary mask (cone between scales)
    mask = sphere(radius=radius, center=centroid)

    # Subsample the whole volume and the mask
    vol_sub, center = extract_sub_volume(vol, radius=radius, centroid=centroid)
    mask_sub, _ = extract_sub_volume(mask, radius=radius, centroid=centroid)

    # Mask sub-sampled volume
    vol_masked_sub = vol_sub * mask_sub

    show_mri(vol_masked_sub, slice_xyz=center)
    plt.suptitle('Original Image')
    plt.show()
