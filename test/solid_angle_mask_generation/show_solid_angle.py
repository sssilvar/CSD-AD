import os
import sys

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# Set root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol


if __name__ == '__main__':
    # a = solid_cone(radius=(5, 40)).astype(np.bool)
    # a.tofile('C:/Users/Smith/Downloads/a.raw')

    img = np.random.normal(128, 100, [256,256]).astype(np.int8)

    print(img.shape)

    img.tofile('C:/Users/Smith/Downloads/a.raw')
    e = np.fromfile('C:/Users/Smith/Downloads/a.raw', dtype=np.int8).reshape([256, 256])
    print(e.shape)

    plt.imshow(e - img, cmap='gray')
    plt.show()
