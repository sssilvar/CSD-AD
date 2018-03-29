import os

import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    # Load image
    img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '0729',
                                'intensity_0_to_25_solid_angle_to_sphere.raw')
    img = np.fromfile(img_filename, dtype=np.float).reshape([360, 180]).T

    # Load PNG
    # img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '30_to_60_solid_angle_to_sphere.png')
    # img = cv2.imread(img_filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = feature.canny(img, sigma=5)
    hog = feature.hog(img)
    print(edges.shape)

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()

