import os
import sys

import pickle
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
    data = np.load(filename).item()
    # with open(filename, 'rb') as fp:
    #     data = pickle.load(fp)

    for key, value in data.items():
        print('Key: %s' % key)
        for el in value:
            print(float(el))

    curvelet_plot(n_scales, n_angles, data)
    plt.show()
