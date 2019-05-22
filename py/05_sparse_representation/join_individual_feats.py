#!/bin/env python2
import os
from os.path import join

import numpy as np
import pandas as pd


# Set folders
data_folder = '/home/jullygh/sssilvar/Documents/code/CSD-AD/output/curv_feats_gradient_nscales_4_nangles_4'
output_file = '/home/jullygh/sssilvar/Documents/code/CSD-AD/output/curv_feats_gradient_nscales_4_nangles_4.csv'
ext = '.npz'

# Load individual feature files
feat_files = [f for f in os.listdir(data_folder) if ext in f]

feat_series = []
for feat_file in feat_files:
    # Create a stack of data Series: feat_series
    data = pd.Series(dict(np.load(join(data_folder, feat_file))))
    data.name = str(data['subject'])
    feat_series.append(data)

# Create final DataFrame: df
df = pd.DataFrame(feat_series)
df.to_csv(output_file)