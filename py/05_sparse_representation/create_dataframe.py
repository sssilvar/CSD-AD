import os
from glob import glob
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))  # root folder


if __name__ == '__main__':
    out_folder = join(root, 'output')
    scales = [5, 6, 7, 9]
    angles = [8, 16, 32]
    img_type = 'gradient'  # Or 'intensity

    for scale in scales:
        print('\n[  OK  ] Processing scale %d' % scale)
        for angle in angles:
            folder = join(out_folder, 'curv_feats_%s_nscales_%d_nangles_%d' % (img_type, scale, angle))
            if os.path.exists(folder):
                # Get *npy files
                npz_files = [npf for npf in glob(join(folder, '*.npz'))]
                print('Number of subjects found: %d' % len(npz_files))
            else:
                print('[  ERROR  ] Folder not found')