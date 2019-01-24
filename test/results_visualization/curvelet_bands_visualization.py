#!/bin/env python2
import os
import sys
from configparser import ConfigParser
from os.path import dirname, join, realpath

import pyct as ct
import numpy as np
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(root)

from lib.curvelets import get_sub_bands

if __name__ == "__main__":
    # Parse configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config/config.cfg'))
    mapped_dir = cfg.get('dirs', 'sphere_mapping')
    
    # Ge a single image
    img_file = join(mapped_dir, '002_S_0729/gradient_0_to_25_solid_angle_to_sphere.raw')
    img = np.fromfile(img_file).reshape([180,90]).T

    # Decompose in curvelets
    nbs = 4
    nba = 16
    A = ct.fdct2(
        (90,90), 
        nbs=nbs, 
        nba=nba, 
        ac=True, 
        norm=False, 
        vec=True, 
        cpx=False)
    f = A.fwd(img)
    curv_data = get_sub_bands(A, f)
    n = len(curv_data)
    ncols = 8
    fig, ax = plt.subplots(ncols=ncols, nrows=np.ceil(n/ncols).astype(int) + 1)
    for i, (key, val) in enumerate(curv_data.items()):
        ix = i // ncols
        for j in range(ncols):
            print(ix, j)
            ax[ix, j].imshow(val)
            ax[ix, j].set_title(key)
            ax[ix, j].axis('off')
    plt.show()



