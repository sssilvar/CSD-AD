import os
import numpy as np
import pandas as pd
import nibabel.freesurfer as nbfs


class Freesurfer:

    columns = ['region_id', 'region_name', 'mean', 'std', 'hemisphere']

    def __init__(self, dataset_folder, df):
        self.dataset_folder = dataset_folder
        self.df = df

    def read_thickness(self, subject_id, hemi):
        # Set subject folder
        subject_folder = os.path.join(self.dataset_folder, subject_id)
        print('[  FS  ] Loading subject %s' % subject_folder)

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

            tk_stats.loc[i] = [i, labels[i], tk_mean, tk_std, hemi]

        morph_data = {
            # 'tk_stats_mean': tk_stats.pivot(index=i, columns=labels, values='mean'),
            # 'tk_stats_std': tk_stats.pivot(index=i, columns=labels, values='std'),
            'ctab': ctab,
            'labels': labels
        }

        return tk_stats, morph_data
