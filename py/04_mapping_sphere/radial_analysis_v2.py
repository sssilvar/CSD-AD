import os
import sys

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import affine_transform

# Set root folder
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Append path to add lib
sys.path.append(root)
from lib.param import load_params
from lib.masks import solid_cone
from lib.visualization import show_mri
from lib.transformations import rotate_vol
from lib.geometry import extract_sub_volume, get_centroid


if __name__ == '__main__':
    print('TODO')
