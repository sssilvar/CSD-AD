import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb

# Choose the file to be played with
filename = 'test_data/941_S_1363.mgz'

# Correct filename
filename = os.path.join(os.getcwd(), filename)
print('[  OK  ] File to be processed is located in: %s' % filename)

# Load MRI file
mri = nb.load(filename)
img = mri.get_data()

# Generate a sphere
r = 50
cx, cy, cz = (20, 20, 20)

ax_min, ax_max = (-128, 128)
y, x, z = np.ogrid[ax_min:ax_max, ax_min:ax_max, ax_min:ax_max]
eqn = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
mask = np.bitwise_and(eqn < (r + 2. / 256 * r) ** 2, eqn > (r - 2. / 256 * r) ** 2)
img_sph = img * mask
img_ball = img * (eqn <= r ** 2)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mask[:, :, 128+49], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_ball[:, :, 128], cmap='gray')
plt.axis('off')

plt.show(block=False)
