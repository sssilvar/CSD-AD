import os

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '0729',
                            'intensity_0_to_25_solid_angle_to_sphere.png')

img = imread(img_filename)
img = rgb2gray(img).T
print(img.shape)

# Right Hemisphere
s = np.linspace(0, 2*np.pi, 400)
x_rh = 70 + 40 * np.cos(s)
y_rh = 120 + 40 * np.sin(s)
init_rh = np.array([x_rh, y_rh]).T

snake_right = active_contour(gaussian(img, 0.8),
                             init_rh, alpha=0.015, beta=10, gamma=0.001)

print(snake_right)

# Left Hemisphere
x_lh = 70 + 220 + 40 * np.cos(s)
y_lh = 120 + 40 * np.sin(s)
init_lh = np.array([x_lh, y_lh]).T

snake_left = active_contour(gaussian(img, 0.8),
                            init_lh, alpha=0.015, beta=10, gamma=0.001)

# Plot results
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)

ax.plot(init_rh[:, 0], init_rh[:, 1], '--r', lw=3)
ax.plot(snake_right[:, 0], snake_right[:, 1], '-b', lw=3)

ax.plot(init_lh[:, 0], init_lh[:, 1], '--r', lw=3)
ax.plot(snake_left[:, 0], snake_left[:, 1], '-b', lw=3)

ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()
