import os

import numpy as np
import pywt

import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def plot_wavelet_decomposition(image, level=3, wavelet='haar'):
    """
    Plot of 2D wavelet decompositions for given number of levels.

    image needs to be either a colour channel or greyscale image:
        rgb: self.I[:, :, n], where n = {0, 1, 2}
        greyscale: use rgb_to_grey(self.I)

    """
    plot_image = np.zeros_like(image)
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        if i == 0:
            cAcH = np.concatenate((coeffs[0] / np.max(coeffs[0]), cH), axis=1)
            cVcD = np.concatenate((cV, cD), axis=1)
            plot_image = np.concatenate((cAcH, cVcD), axis=0)
        else:
            plot_image = np.concatenate((plot_image, cH), axis=1)
            cVcD = np.concatenate((cV, cD), axis=1)
            plot_image = np.concatenate((plot_image, cVcD), axis=0)

    plt.figure()
    plt.grid(False)
    plt.imshow(abs(plot_image), cmap='gray')
    plt.title('Wavelet (%s) decomposition - Level %d' % (wavelet, level))
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()


if __name__ == '__main__':
    img_filename = os.path.join(root, 'test', 'test_data', 'sphere_mapped', '0729',
                                'intensity_0_to_25_solid_angle_to_sphere.raw')
    img = np.fromfile(img_filename, dtype=np.float).reshape([360, 180]).T

    plot_wavelet_decomposition(img, level=2, wavelet='haar')
    plt.show()
