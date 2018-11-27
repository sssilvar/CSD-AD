#!/bin/env python
import os
import sys
from os.path import join, basename, dirname, realpath

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Define root folder
root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == "__main__":
    # Load dataset
    data_file = '~/Documents/temp/results/curvelet_roi_feats/curvelet_features_rois_4_scales_4_angles.csv'
    # data_file = sys.argv[1]

    print('[  INFO  ] Loading dataset...')
    df = pd.read_csv(data_file, index_col=0)
    df['ROI'] = df['ROI'].astype('category')

    # TODO: Rest of the code

