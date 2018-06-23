from __future__ import print_function

import os

import numpy as np
import pandas as pd


up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))


if __name__ == '__main__':
    print('Starting analysis')
    features_folder = os.path.join(root, 'features', 'curvelets')
    csv_data = os.path.join(root, 'param', 'data_df.csv')
    regions = os.path.join(root, 'param', 'FreeSurferColorLUT.csv')

    n_comp = [3, 7, 11]

    # Read datas into pandas DataFrame
    df = pd.read_csv(csv_data)
    df_regions = pd.read_csv(regions, index_col=['region_id'])

    rois = []
    for comp in n_comp:
        csv_output = os.path.join(features_folder, 'curvelet_gmm_%d_comp.csv' % comp)
        if not os.path.exists(csv_output):
            for i, subject in enumerate(df['folder']):
                print('[  INFO  ] Processing subject: ', subject)
                feature_file = os.path.join(features_folder, subject, 'curvelet_gmm_%d_comp.npy' % comp)

                if i == 0:
                    print('[  INFO  ] Creating dataframe')
                    df_features = pd.DataFrame(np.load(feature_file).item(), index=[0])
                else:
                    subject_features = pd.DataFrame(np.load(feature_file).item(), index=[0])
                    df_features = df_features.append(subject_features)
            # Save csv
            df_features.to_csv(csv_output)
        else:
            print('Reading CSV ...')
            df = pd.read_csv(csv_output, index_col=0)

            print('Droping useless columns...')
            df = df.dropna(axis=1)
            df.to_csv(csv_output)
            print('Done!')

    print('DONE!\n\n')
