from __future__ import print_function

import os
import numpy as np
import pandas as pd
import nibabel as nb
import nibabel.freesurfer as nbfs
import matplotlib.pyplot as plt
import json

# Load params
with open('param/params.json') as json_file:
    jf = json.load(json_file)
    dataset_folder = jf['dataset_folder']
    data_file = jf['data_file']

# Load DataFrame
df = pd.read_csv(data_file)

# Load random subject
i = np.random.randint(1, 10)
img_dir = os.path.join(dataset_folder, df['folder'].iloc[i], 'mri', 'brainmask.mgz')
img = nb.load(img_dir)
print('[  OK  ] Image loaded')

# Load annotations
annot_path = os.path.join(dataset_folder, df['folder'].iloc[i], 'label', 'lh.aparc.annot')
annot = nbfs.read_annot(annot_path)
print('[  OK  ] Annotations file loaded')

# Load thickness
tk_path = os.path.join(dataset_folder, df['folder'].iloc[i], 'surf', 'lh.thickness')
tk = nbfs.read_morph_data(tk_path)
print('[  OK  ] Thickness file loaded')

#