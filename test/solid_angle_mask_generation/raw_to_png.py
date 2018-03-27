import os

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    raw_file = os.path.join(root, 'output', '66_to_100_solid_angle_to_sphere.raw')
    arr = np.fromfile(raw_file).reshape([360, 180])

    plt.subplot(121)
    plt.imshow(arr, cmap='gray')
    plt.subplot(122)
    plt.hist(arr)
    plt.show()
