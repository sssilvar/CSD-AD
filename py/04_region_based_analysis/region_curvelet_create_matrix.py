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

    n_comp = [3]

    # Read datas into pandas DataFrame
    df = pd.read_csv(csv_data)
    df_regions = pd.read_csv(regions, index_col=['region_id'])

    rois = []
    for comp in n_comp:
        for subject in df['folder'][:2]:
            feature_file = os.path.join(features_folder, subject, 'curvelet_gmm_%d_comp.csv.npy' % comp)
            subject_features = np.load(feature_file).item()
            rois.append(subject_features.keys())

    print(len(rois[0]))
    print(len(rois[1]))

    matchingA = list(set(rois[0]) - set(rois[1]))
    matchingB = list(set(rois[1]) - set(rois[0]))

    print(matchingA)
    print(matchingB)