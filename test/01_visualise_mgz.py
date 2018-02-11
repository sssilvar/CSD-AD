from __future__ import print_function

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# Assign images directory
# filename = os.path.join(os.getcwd(), 'test_data', 'fsaverage.mgz')
filename = '/run/media/sssilvar/DATA/FreeSurferSD/002_S_0729/mri/aseg.mgz'
slide = 128

# Load image
mgz = nb.load(filename)
img = mgz.get_data()
print('[ OK ] Image shape:\t', img.shape)

# Set plot up
plt.style.use('ggplot')
plt.figure()
plt.title('MGZ slide %d view' %slide)
plt.imshow(img[:, :, slide], cmap='gray')
plt.axis('off')
plt.show()

