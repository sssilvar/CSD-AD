#!/bin/env python3
import os
from configparser import ConfigParser
from os.path import join, dirname, realpath, basename

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.style.use('ggplot')


def imread(filename):
    if filename.endswith('.raw'):
        return np.fromfile(filename).reshape([180, 90]).T
    else:
        return imageio.imread(filename)[:,:,0].T


def get_dx_age_and_size(df):
    dx = df.loc[sid, 'target']
    age = df.loc[sid, 'AGE']
    sex = df.loc[sid, 'PTGENDER']
    return dx, age, sex



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
    im_type = 'gradient'
    images = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f == '{}_{}_solid_angle_to_sphere.raw'.format(im_type, scale):
                im_path = join(root, f)
                images.append(im_path)
    
    # Plot images
    n_cols = 5
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows)
    # fig.suptitle('Sphere mapped visualization ({} images)'.format(im_type))
    fig.subplots_adjust(wspace=0, hspace=0.4)

    for i, axs in enumerate(axes):
        if np.shape(axs):
            for j, ax in enumerate(axs):
                try:
                    img_file = images[int((n_cols * i) + j)]
                    img = imread(img_file)
                    ax.imshow(img, cmap='gray')

                    # Set title
                    sid = basename(dirname(img_file))
                    tg, age, sex = get_dx_age_and_size(df)
                    ax.set_title('{} | {} yrs | {}'.format(tg, age, sex), color='r' if tg == 'MCIc' else 'b')

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
                tg, age, sex = get_dx_age_and_size(df)
                axs.set_title('{} | {} yrs | {}'.format(tg, age, sex), color='r' if tg == 'MCIc' else 'b')
            except IndexError as e:
                print('[ WARNING  ] {}'.format(e))
            
            # Extra config
            axs.axis('off')
    plt.show()
