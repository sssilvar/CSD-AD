import os
import numpy as np
import matplotlib.pyplot as plt

# Set root folder
root = os.path.join(os.getcwd(), '..', '..')

if __name__ == '__main__':
    ax_min, ax_max = (-5, 5)
    x, y, z = np.ogrid[ax_min:ax_max, ax_min:ax_max, ax_min:ax_max]

    theta = np.floor(np.arctan(y / x) * 10) == np.floor(np.pi/4 * 10)
    l = np.bitwise_and(theta, z == z)

    img = np.ones([10, 10, 10]) * l
    print(l.shape)
    print(img.shape)

    plt.figure()
    plt.imshow(img[:, :, 9], cmap='gray')
    plt.show(block=False)
