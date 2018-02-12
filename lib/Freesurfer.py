import os
import numpy as np
import pandas as pd
import nibabel.freesurfer as nbfs


class Freesurfer:

    # columns = ['region_id', 'region_name', 'mean', 'std', 'hemisphere']

    def __init__(self, dataset_folder, df):
        self.dataset_folder = dataset_folder
        self.df = df

    def read_thickness(self, subject_id):
        # Set subject folder
        subject_folder = os.path.join(self.dataset_folder, subject_id)
        print('[  FS  ] Loading subject %s' % subject_folder)

        # Load labels
        labels_id_lh, ctab_lh, labels_lh = nbfs.read_annot(os.path.join(subject_folder, 'label', 'lh' + '.aparc.annot'))
        labels_id_lh[labels_id_lh == -1] = 0  # -1 Correction
        thickness_lh = nbfs.read_morph_data(os.path.join(subject_folder, 'surf', 'lh' + '.thickness'))

        labels_id_rh, ctab_rh, labels_rh = nbfs.read_annot(os.path.join(subject_folder, 'label', 'rh' + '.aparc.annot'))
        labels_id_rh[labels_id_rh == -1] = 0  # -1 Correction
        thickness_rh = nbfs.read_morph_data(os.path.join(subject_folder, 'surf', 'rh' + '.thickness'))


        columns = map(lambda x: 'lh_' + x + '_mean', labels_lh) + map(lambda x: 'lh_' + x + '_std', labels_lh) + \
                  map(lambda x: 'rh_' + x + '_mean', labels_rh) + map(lambda x: 'rh_' + x + '_std', labels_rh)
        tk_stats = pd.DataFrame(columns=columns)

        # Left hemisphere
        tk_mean_lh = []
        tk_std_lh = []
        tk_mean_rh = []
        tk_std_rh = []

        for i, label in enumerate(labels_lh):
            if thickness_lh[labels_id_lh == i].any():
                tk_mean_lh.append(np.mean(np.nan_to_num(thickness_lh[labels_id_lh == i])))
                tk_std_lh.append(np.std(np.nan_to_num(thickness_lh[labels_id_lh == i])))
            else:
                tk_mean_lh.append(0)
                tk_std_lh.append(0)

        for i, label in enumerate(labels_lh):
            if thickness_lh[labels_id_lh == i].any():
                tk_mean_rh.append(np.mean(np.nan_to_num(thickness_rh[labels_id_rh == i])))
                tk_std_rh.append(np.std(np.nan_to_num(thickness_rh[labels_id_rh == i])))
            else:
                tk_mean_rh.append(0)
                tk_std_rh.append(0)

        tk_stats.loc[0] = tk_mean_lh + tk_std_lh + tk_mean_rh + tk_std_rh

        morph_data = {
            'labels_id': [labels_id_lh, labels_id_rh],
            'ctab': [ctab_lh, ctab_rh],
            'labels': [labels_lh, labels_rh]
        }

        return tk_stats, morph_data
