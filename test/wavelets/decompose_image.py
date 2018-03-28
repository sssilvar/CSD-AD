import os

import pywt
import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '30_to_60_solid_angle_to_sphere.raw')
    img = np.fromfile(img_filename, dtype=np.float).reshape([360, 180]).T


    # Decompose image
    wp = pywt.WaveletPacket2D(data=img, wavelet='haar', mode='symmetric')
    print(wp.data)
    print(repr(wp.path))
    print(wp.level)
    print(wp.maxlevel)

    # PLot resutls
    plt.figure()
    plt.subplot(221)
    plt.imshow(wp['a'].data, cmap='gray')
    plt.subplot(222)
    plt.imshow(wp['h'].data, cmap='gray')
    plt.subplot(223)
    plt.imshow(wp['v'].data, cmap='gray')
    plt.subplot(224)
    plt.imshow(wp['d'].data, cmap='gray')

    plt.figure()
    plt.imshow(np.sqrt(wp['h'].data ** 2 + wp['v'].data ** 2) * wp['a'].data - wp['d'].data, cmap='gray')

    plt.show()
