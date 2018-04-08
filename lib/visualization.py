import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def show_mri(vol, slice_xyz=(128, 128, 128), axis='off', grid='off'):
    prop = fm.FontProperties(
        fname=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', 'Fira_Sans', 'FiraSans-Bold.ttf')
    )
    fontsize = 18
    color = (0.09, 0.11, 0.18)

    img_x = vol[:, :, slice_xyz[0]]
    img_y = vol[:, slice_xyz[1], :]
    img_z = vol[slice_xyz[2], :, :]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_x, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

    plt.subplot(1, 3, 2)
    plt.imshow(img_y, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('XZ Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

    plt.subplot(1, 3, 3)
    plt.imshow(img_z, cmap='gray')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

