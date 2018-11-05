import os
from os.path import join

import pyct as ct
from imageio import imread

import pandas as pd
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    os.system('clear')
    
    # Load dataset in dataframe
    for i in range(2):
        data_folder = '/home/ssilvari/data/mnist_png/training/%d' % i
        print('[  INFO  ] Loading %s' % data_folder)
        imgs = os.listdir(data_folder)
        imgs = imgs [:5]

        # Load images
        for im in imgs:
            img_file = join(data_folder, im)
            img = imread(img_file)

            print(img.shape)
