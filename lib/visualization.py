import matplotlib.pyplot as plt
import numpy as np


def show_mri(vol, slice_xyz=(128, 128, 128)):
    img_x = vol[:, :, slice_xyz[0]].reshape((256, 256))
    img_y = vol[:, slice_xyz[1], :].reshape((256, 256))
    img_z = vol[slice_xyz[2], :, :].reshape((256, 256))

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_x, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 3, 2)
    plt.imshow(img_y, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(1, 3, 3)
    plt.imshow(img_z, cmap='gray')
    plt.xlabel('Y')
    plt.ylabel('Z')
