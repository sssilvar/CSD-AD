import os
import nibabel as nb
import logging as log
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Define parameters
filename = os.path.join(os.getcwd(), 'test_data', '941_S_1363.mgz')
print(filename)
slide = 128

# Load image
log.warning('Loading image...')
mgz = nb.load(filename)
img = mgz.get_data()

# Apply gaussian filter
sigma = 5
order = 0
img_filtered = ndi.gaussian_filter(img, sigma=sigma, order=order)

# Show image
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img[:, :, slide], cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_filtered[:, :, slide], cmap='gray')
plt.title('Gaussian filter (Sigma %.2f | order %d)' % (sigma, order))
plt.axis('off')
plt.show()
