from __future__ import print_function

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import scipy.special as sp

# Assign images directory
# filename = os.path.join(os.getcwd(), 'test_data', 'fsaverage.mgz')
filename = '/run/media/sssilvar/2A3CA00E3C9FD2E5/Users/Smith/Documents/FreeSurferSD/002_S_0729/mri/brainmask.mgz'
slide = 117

# Load image
mgz = nb.load(filename)
img = mgz.get_data().astype(np.float32)

# Gradient calculation (cartesian coordinates)
x_, y_, z_ = np.gradient(img, edge_order=2)

# Transformation to shperical coordinates
r, theta, phi = np.nan_to_num((
    np.int8(np.sqrt(x_ ** 2 + y_ ** 2 + z_ ** 2)),
    np.tanh(y_ / x_),
    np.tanh(np.sqrt(x_ ** 2 + y_ ** 2) / z_)
))


print('[ OK ] Image shape:\t', img.shape)

# Set plot up
plt.imsave('/home/sssilvar/Pictures/s3.png', r[:,:,slide], cmap='gray')

plt.style.use('ggplot')
plt.figure()
plt.title('MGZ slide %d view' %slide)
plt.imshow(r[:, :, slide], cmap='gray')
plt.axis('off')


# Spherical harmonics calculation
sph_lambda = (lambda r_, theta_, phi_: sp.sph_harm(r_, 2, theta_, phi_))
sh_func = np.vectorize(sph_lambda)
sh_complex = np.nan_to_num(sh_func(r, theta, phi))


# Set plot up
plt.imsave('/home/sssilvar/Pictures/s4.png', sh_complex.__abs__()[:,:,slide])
plt.figure()
plt.title('SGH slide %d view' %slide)
plt.imshow(sh_complex.__abs__()[:, :, slide], cmap='gray')
plt.axis('off')

plt.show()

