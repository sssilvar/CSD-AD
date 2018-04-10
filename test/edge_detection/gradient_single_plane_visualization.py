import os

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':
    filename = os.path.join(root, 'test', 'test_data', '941_S_1363.mgz')
    mgz = nb.load(filename)
    img = mgz.get_data().astype(np.float)

    # Calculate gradients
    x, y, z = np.gradient(img)
    mag = np.sqrt(x**2+y**2+z**2)

    # Plot results (single plane)
    plt.imshow(mag[:,125,:], cmap='gray')
    plt.axis('off')
    plt.show()
