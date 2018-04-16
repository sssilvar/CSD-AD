import os
import sys

import json
import numpy as np
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(root))

from lib.curvelets import curvelet_plot

if __name__ == '__main__':
    filename = sys.argv[1]
    n_scales = sys.argv[2]
    n_angles = sys.argv[3]

    print('Loading %s...' % filename)
    with open(filename) as fp:
        data = json.load(fp)

    curvelet_plot(n_scales, n_angles, data)
    plt.show()
