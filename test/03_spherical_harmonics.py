import os
from math import pi
import numpy as np
import nibabel as nb
import logging as log
import scipy.special as sp
import matplotlib.pyplot as plt

# Define parameters
filename = os.path.join(os.getcwd(), 'test_data', '941_S_1363.mgz')
print(filename)
slide = 3

# Load image
log.warning('Loading image...')
mgz = nb.load(filename)
img = mgz.get_data().astype('float64')
img = img[:, :, 125:130]

gx, gy, gz = np.gradient(img)
r, theta, phi = np.nan_to_num((
    np.sqrt(gx ** 2 + gy ** 2 + gz ** 2),
    np.arctan(gy / gx),
    np.arctan(np.sqrt(gx ** 2 + gy ** 2) / gz)
))

plt.imshow(r[:, :, slide], cmap='gray')

# Calculate SH
sph_lambda = (lambda r, theta, phi: sp.sph_harm(r, 5, theta, phi))

sh_func = np.vectorize(sph_lambda)
sh = sh_func(r, theta, phi)
sh_mag = np.nan_to_num(np.abs(sh))
sh_mag_norm = (sh_mag / sh_mag.max())
print(sh)

# Plot
plt.figure()
plt.imshow(img[:, :, slide], cmap='gray')
plt.figure()
plt.imshow(sh_mag_norm[:, :, slide], cmap='gray')
plt.show(block=False)
