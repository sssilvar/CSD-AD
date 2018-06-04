import os
import sys

import numpy as np
from skimage.io import imread
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(root)
from lib.geometry import solid_cone, sphere
from lib.transformations import rotate_vol
from lib.visualization import show_mri
from lib.geometry import get_centroid, extract_sub_volume

plt.style.use('ggplot')


def show_4slices(vol, slice_xyz=(128, 128, 128), axis='off', grid='off', render=''):
    prop = fm.FontProperties(
        fname=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts', 'Fira_Sans', 'FiraSans-Bold.ttf')
    )
    fontsize = 18
    color = (0.09, 0.11, 0.18)

    img_x = vol[:, :, slice_xyz[0]].T
    img_y = vol[:, slice_xyz[1], :].T
    img_z = vol[slice_xyz[2], :, :]

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_x, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title('XY Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

    plt.subplot(2, 2, 2)
    plt.imshow(img_y, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Z')
    # plt.title('XZ Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

    plt.subplot(2, 2, 3)
    plt.imshow(img_z, cmap='gray')
    plt.xlabel('Y')
    plt.ylabel('Z')
    # plt.title('YZ Plane', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)

    plt.subplot(2, 2, 4)
    plt.imshow(render)
    plt.xlabel('Y')
    plt.ylabel('Z')
    # plt.title('3D View', fontproperties=prop, size=fontsize, color=color)
    plt.axis(axis)
    plt.grid(grid)


if __name__ == '__main__':
    # Define file names
    vol_filename = os.path.join(root, 'test', 'test_data', 'brainmask.mgz')
    aseg_filename = os.path.join(root, 'test', 'test_data', 'mri', 'aseg.mgz')
    render_dir = os.path.join(root, 'output', 'render')

    # Load images
    vol = nb.load(vol_filename).get_data()
    aseg = nb.load(aseg_filename).get_data()

    # Define radius and center
    # radius = (0, 25)
    scales = [
        (0, 25),
        (25, 50),
        (50, 75),
        (75, 100)
    ]
    centroid = tuple(get_centroid(aseg > 0))
    print('[  OK  ] Centroid = {}'.format(centroid))

    for i, radius in enumerate(scales):
        # Create a binary mask (cone between scales)
        mask = sphere(radius=radius, center=centroid)

        # Subsample the whole volume and the mask
        vol_sub, center = extract_sub_volume(vol, radius=radius, centroid=centroid)
        mask_sub, _ = extract_sub_volume(mask, radius=radius, centroid=centroid)

        mask_sub = rotate_vol(mask_sub, angles=(45, 0, 0))
        solid_ang_mask = rotate_vol(mask_sub, angles=(0, 0, -180))

        # Mask sub-sampled volume
        vol_masked_sub = vol_sub * solid_ang_mask
        mean_int = np.nan_to_num(vol_masked_sub.sum() / solid_ang_mask.sum())

        print(mean_int)
        image_out_filename = os.path.join(root, 'output', 'intensity_%d_to_%d_three_views.png' % radius)

        # Load render
        # img = imread(os.path.join(render_dir, '%d.png' % (i + 3)))
        img = vol[:, :, 128].T

        # Plot them all
        show_4slices(vol_masked_sub, slice_xyz=center, render=img)
        # show_mri(vol_masked_sub, slice_xyz=center)
        # plt.savefig(image_out_filename, cmap='gray', bbox_inches='tight')
        plt.show()
