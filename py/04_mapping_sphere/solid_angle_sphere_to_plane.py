import os
import sys

import matplotlib.pyplot as plt

# Set root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.masks import solid_cone
from lib.visualization import show_mri


if __name__ == '__main__':
    a = solid_cone(radius=(0, 33))

    show_mri(a, slice_xyz=(128, 128, 128))

    plt.show()
