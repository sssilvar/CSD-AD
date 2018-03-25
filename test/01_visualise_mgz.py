from __future__ import print_function

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import scipy.special as sp

root = os.path.dirname(os.path.dirname(__file__))

# Assign images directory
filename = os.path.join(os.getcwd(), 'test_data', 'fsaverage.mgz')
# filename = 'C:/Users/Smith/Downloads/temp/gradients/002_S_1070/phi.mgz'
slide = 117

# Load image
mgz = nb.load(filename)
img = mgz.get_data()

print('[ OK ] Image shape:/t', img.shape)


plt.style.use('ggplot')
plt.figure()
plt.title('MGZ slide %d view' %slide)
plt.imshow(img[:, :, slide], cmap='gray')
plt.axis('off')
plt.show()
