from __future__ import print_function

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load folder
folder = '/home/sssilvar/PycharmProjects/CSD-AD/features/left_hippo'

# Plot images
MCIc_folder = os.path.join(folder, 'MCIc')
MCInc_folder = os.path.join(folder, 'MCInc')

a = os.listdir(MCIc_folder)
b = os.listdir(MCInc_folder)

out_folder = '/home/sssilvar/PycharmProjects/CSD-AD/features/left_hippo/plot'


# List files
for i in range(18):
    # plt.figure()
    # plt.suptitle('SH magnitude - L-Hippo')

    plt.subplot(1, 2, 1)
    plt.title('MCIc')
    img1 = mpimg.imread(os.path.join(MCIc_folder, a[i]))
    plt.imshow(img1)

    plt.subplot(1, 2, 2)
    plt.title('MCInc')
    img2 = mpimg.imread(os.path.join(MCInc_folder, b[i]))
    plt.imshow(img2)

    plt.savefig(os.path.join(out_folder, 'png', '%s.png' % i), bbox_inches='tight')
