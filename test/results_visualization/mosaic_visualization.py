#!/bin/env python3
import os
from configparser import ConfigParser
from os.path import join, dirname, realpath, basename

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def imread(filename):
    if filename.endswith('.raw'):
        return np.fromfile(filename).reshape([360, 180]).T
    else:
        return imageio.imread(filename)[:,:,0].T


if __name__ == "__main__":
    # Define root folder, and parse configurations
    root = dirname(dirname(dirname(realpath(__file__))))
    cfg_file = join(root, 'config', 'config.cfg')
    cfg = ConfigParser()
    cfg.read(cfg_file)

    data_folder = cfg.get('dirs', 'sphere_mapping')

    # Load ADNI data
    df = pd.read_csv(join(root, 'param/df_conversions.csv'), index_col='PTID')

    # Look for images
    scale = '0_to_25'
    im_type = 'intensity'
    images = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f == '{}_{}_solid_angle_to_sphere.raw'.format(im_type, scale):
                im_path = join(root, f)
                images.append(im_path)
    
    # Plot images
    n_cols = 3
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    fig.tight_layout()
    fig.suptitle('Sphere mapped visualization ({} images)'.format(im_type))

    for i, axs in enumerate(axes):
        if np.shape(axs):
            for j, ax in enumerate(axs):
                try:
                    img_file = images[int((n_cols * i) + j)]
                    img = imread(img_file)
                    ax.imshow(img, cmap='gray')

                    # Set title
                    sid = basename(dirname(img_file))
                    tg = df.loc[sid, 'target']
                    ax.set_title('{}'.format(tg), color='r' if tg == 'MCIc' else 'b')

                except IndexError as e:
                    print('[ WARNING  ] {}'.format(e))
                
                # Extra config
                ax.axis('off')
        else:
            try:
                img_file = images[i]
                img = imread(img_file)
                axs.imshow(img, cmap='gray')

                # Set title
                sid = basename(dirname(img_file))
                tg = df.loc[sid, 'target']
                axs.set_title('{}'.format(tg), color='r' if tg == 'MCIc' else 'b')
            except IndexError as e:
                print('[ WARNING  ] {}'.format(e))
            
            # Extra config
            axs.axis('off')
    plt.show()
