#!/bin/env python3
import os
from configparser import ConfigParser
from os.path import join, dirname, realpath, basename

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# plt.style.use('ggplot')


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
    scale = '25_to_50'
    im_type = 'gradient'
    images = []
    for root, dirs, files in os.walk(data_folder):
        for f in files:
            if f == '{}_{}_solid_angle_to_sphere.raw'.format(im_type, scale):
                im_path = join(root, f)
                images.append(im_path)
    
    # images = images[:30]
    # Plot images
    print('Number of subjects found: {}'.format(len(images)))
    n_cols = 6
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
                    # ax.set_title('{} | {} yrs | {}'.format(tg, age, sex), color='r' if tg == 'MCIc' else 'b')
                    ax.set_title('{} '.format(tg), color='r' if tg == 'MCIc' else 'b')


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
    # plt.savefig('/tmp/plot.pdf', bbox_inches='tight', orientation='landscape', papertype='a0')
    # plt.show()

    # Mean absolute difference
    mcic_img = np.zeros_like(img)
    mcinc_img = np.zeros_like(mcic_img)
    mcic_count = 0
    mcinc_count = 0

    for img_file in images:
        sid = basename(dirname(img_file))
        img = imread(img_file)
        
        dx = df.loc[sid, 'target']
        if dx == 'MCIc':
            mcic_img += img
            mcic_count += 1
        elif dx == 'MCInc':
            mcinc_img += img
            mcinc_count += 1
    
    # Do average
    mcic_img /= mcic_count
    mcinc_img /= mcinc_count
    
    fig, ax = plt.subplots(ncols=3, nrows=1)
    ax[0].imshow(mcic_img)
    ax[1].imshow(mcinc_img)
    ax[2].imshow(np.abs(mcic_img - mcinc_img))
    plt.show()
