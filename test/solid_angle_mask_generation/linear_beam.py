#!/bin/env python3
import numpy as np


def ray_trace(rmin, rmax, center=(128, 128, 128), shape=(256, 256, 256)):
    # Parse shape and center
    sx, sy, sz = shape
    cx, cy, cz = center

    # Define grid and re-center it
    x, y, z = np.ogrid[0:sx, 0:sy, 0:sz]
    x, y, z = (x - cx, y - cy, z - cz)

    # Define a cone with squared section
    eqn_rays = x - y


if __name__ == '__main__':
    print('test')
