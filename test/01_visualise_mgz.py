from __future__ import print_function

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import scipy.special as sp
from lib.visualization import show_mri

root = os.path.dirname(os.path.dirname(__file__))

# Assign images directory
# filename = os.path.join(os.getcwd(), 'test_data', 'fsaverage.mgz')
filename = "Z:/Users/Smith/Downloads/mni_icbm152_nlin_sym_09c_minc2/mni_icbm152_t1_tal_nlin_sym_09c.mnc"
slide = 117

# Load image
mgz = nb.load(filename)
img = mgz.get_data()
print('MAX Intensity: %d' % img.max())
print('Shape of the image: {}'.format(img.shape))
print('[ OK ] Image shape:/t', img.shape)


plt.style.use('ggplot')
show_mri(img)
plt.suptitle('MGZ slide %d view' %slide)
plt.show()
