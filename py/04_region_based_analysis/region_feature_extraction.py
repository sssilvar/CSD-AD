from __future__ import print_function
import os

import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

plt.style.use('ggplot')

up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

if __name__ == '__main__':
    print('Starting analysis')
    # dataset_folder = '/run/media/ssilvari/HDD Data/Universidad/MSc/Thesis/Dataset/FreeSurferSD'
    dataset_folder = '/home/jullygh/sssilvar/Documents/Dataset/FreeSurferSD'
    csv_data = os.path.join(root, 'param', 'data_df.csv')
    regions = os.path.join(root, 'param', 'FreeSurferColorLUT.csv')

    df = pd.read_csv(csv_data)
    df_regions = pd.read_csv(regions, index_col=['region_id'])

    # Number of components for GMM
    n_comp = 6

    # Features
    feature_list = []

    # Start feature extraction
    for subject in df['folder']:
        print('\n\n[  INFO  ] Processing subject: ', subject)

        # Wipe feature dict
        features_subj = {}

        # Define filenames
        brainmask = os.path.join(dataset_folder, subject, 'mri', 'brainmask.mgz')
        aseg = os.path.join(dataset_folder, subject, 'mri', 'aseg.mgz')

        # Load images
        brainmask = nb.load(brainmask).get_data()
        aseg = nb.load(aseg).get_data()

        # Calculate gradients
        gx, gy, gz = np.gradient(brainmask)
        r, theta, phi = (np.sqrt(gx ** 2 + gy ** 2 + gz ** 2),
                         np.arctan2(gy, gx),
                         np.arctan2(np.sqrt(gx ** 2 + gy ** 2), gz))

        # Initialize individual feature_vector
        features_subj['sid'] = subject
        features_subj['target'] = df['dx_group'].loc[df['folder'] == subject].values[0]
        print('\tDiagnostic group: ', features_subj['target'])

        for region in df_regions.index:
            roi_name = df_regions['label_name'].loc[region]

            # Feature space per region (intensities and gradients')
            roi_feat_space = {
                'int': None,
                'r': None,
                'theta': None,
                'phi': None
            }

            for key in roi_feat_space.keys():
                # look for non-existing regions
                if not np.any(aseg == region):
                    print('[  WARNING  ] ROI %s not found (%s)' % (roi_name, key))

                    # Save null results
                    for i in range(n_comp):
                        features_subj[key + 'mean_' + roi_name + '_' + str(i)] = np.nan
                        features_subj[key + 'cov_' + roi_name + '_' + str(i)] = np.nan
                elif region == 0:
                    print('[  WARNING  ] Unknown region ignored')
                else:
                    ix = np.where(aseg == region)

                    # Feature space per region (intensities and gradients')
                    roi_feat_space = {
                        'int': brainmask[ix].ravel(),
                        'r': r[ix].ravel(),
                        'theta': theta[ix].ravel(),
                        'phi': phi[ix].ravel()
                    }
                    try:
                        # Fit GMM
                        gmm = GaussianMixture(n_components=n_comp)

                        gmm = gmm.fit(X=np.expand_dims(roi_feat_space[key], 1))
                        for i in range(n_comp):
                            features_subj[key + '_mean_' + roi_name + '_' + str(i)] = gmm.means_[i, 0]
                            features_subj[key + '_cov_' + roi_name + '_' + str(i)] = gmm.covariances_[i, 0, 0]

                        # # Evaluate and visualize GMM
                        # gmm_x = np.linspace(0, 253, 256)
                        # gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
                        #
                        # plt.hist(feature, 255, [2, 256], density=True, color='b')
                        # plt.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
                        # plt.title('ROI (' + key + '): ' + roi_name)
                        # plt.show()

                        print('[  OK  ] Region %s successfully processed (%s)' % (roi_name, key))
                    except ValueError as e:
                        print('[  ERROR  ] ', e, ' | ROI voxels: ', len(roi_feat_space[key]), ' | ROI : ', roi_name)

        feature_list.append(features_subj)

    # print(features)
    # Create a DataFrame
    df_features = pd.DataFrame(feature_list)
    df_features.to_csv(os.path.join(root, 'features', 'gmm_features.csv'))
