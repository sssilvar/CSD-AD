#!/usr/bin/env python
# title           :main.py
# description     :This will create a header for a python script.
# author          :Santiago S. Silva R.
# date            :20171112
# version         :0.1
# usage           :python main.py
# notes           :
# python_version  :2.7
# ==============================================================================

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# PARAMETERS
nii_fn = os.path.join('/home/sssilvar/Pictures/NIFTI/test/test.nii')

# FUNCTIONS (DO NOT TOUCH)
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')


def cart_to_spher(x, y, z):
    """Converts from cartesian to spherical coordinates"""
    mag = np.dot(x, y)
    print mag.shape()

    return mag

# Load image
nii = nb.load(nii_fn)
img = nii.get_data().squeeze()

ig_x, ig_y, ig_z = np.gradient(img.astype(float))
print 'Gradient Shape: ', np.shape(ig_x), '|', np.shape(ig_y), '|', np.shape(ig_z)

grad = ig_x
slice_1 = grad[128, :, :]
slice_2 = grad[:, 128, :]
slice_3 = grad[:, :, 128]

c2 = np.dot(ig_y, ig_z)

show_slices([slice_1, slice_2, slice_3])

plt.show()
