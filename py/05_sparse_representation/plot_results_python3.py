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
    n_scales = int(sys.argv[2])
    n_angles = int(sys.argv[3])

    # Set output folder [curvelets/png]
    # Here, the visual results will be saved
    output_folder = os.path.join(os.path.dirname(filename), 'png')
    filename_output = os.path.join(output_folder, os.path.basename(filename)[:-4] + '.png')

    print('Loading %s...\n\n' % filename)
    data = np.load(filename).item()

    print('Saving figure at: ' + filename_output)
    curvelet_plot(n_scales, n_angles, data)
    plt.show()
