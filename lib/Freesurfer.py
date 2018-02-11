import os
import numpy as np
import pandas as pd
import nibabel.freesurfer as nbfs


class Freesurfer:

    columns = ['region_id', 'region_name', 'mean', 'std']

    def __init__(self, dataset_folder, df):
        self.dataset_folder = dataset_folder
        self.df = df

    def read_thickness(self, i, hemi):
        # Set subject folder
        subject_folder = os.path.join(self.dataset_folder, self.df['folder'][i])

        # Load labels
        labels_id, ctab, labels = nbfs.read_annot(os.path.join(subject_folder, 'label', hemi + '.aparc.annot'))
        labels_id[labels_id == -1] = 0  # -1 Correction
        thickness = nbfs.read_morph_data(os.path.join(subject_folder, 'surf', hemi + '.thickness'))

        tk_stats = pd.DataFrame(columns=self.columns)
        for i, label in enumerate(labels):
            if thickness[labels_id == i].any():
                tk_mean = np.mean(np.nan_to_num(thickness[labels_id == i]))
                tk_std = np.std(np.nan_to_num(thickness[labels_id == i]))
            else:
                tk_mean = 0
                tk_std = 0

            tk_stats.loc[i] = [i, labels[i], tk_mean, tk_std]

        morph_data = {
            'labels_id': labels_id,
            'ctab': ctab,
            'labels': labels
        }

        return tk_stats, morph_data
