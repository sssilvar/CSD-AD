import os
import pandas as pd
import numpy as np
import nibabel as nb
import nibabel.freesurfer as nbfs
import lib.Freesurfer as Fs

# Set dataset directory
dataset_folder = '/run/media/sssilvar/DATA/FreeSurferSD'  # Desktop
root = os.path.join(os.getcwd(), '..')
dataset_file = os.path.join(root, 'param', 'data_df.csv')

# Load dataset data into a DataFrame: df
df = pd.read_csv(dataset_file)

# Define a subject
i = 10
subject_folder = os.path.join(dataset_folder, df['folder'][i])

# Load labels
Fs = Fs.Freesurfer(dataset_folder, df)
tk_stats_lh, _ = Fs.read_thickness(i, 'lh')
tk_stats_rh, _ = Fs.read_thickness(i, 'rh')


#