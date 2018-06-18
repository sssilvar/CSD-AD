from __future__ import print_function
import os

import pyct as ct
import numpy as np
import pandas as pd
import nibabel as nb
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

plt.style.use('ggplot')

up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))


def get_cl_info(f, scale, angle):
    try:
        data = f(scale, angle)
        print('Scale %d / angle %d: ' % (scale, angle),
              '\t Shape: ', np.shape(data))
        return np.array(data)
    except IndexError:
        print('[  ERROR  ] Index (%d, %d) out of range' % (scale, angle))


if __name__ == '__main__':
    print('Starting analysis')
    # dataset_folder = '/run/media/ssilvari/HDD Data/Universidad/MSc/Thesis/Dataset/FreeSurferSD'
    # dataset_folder = '/media/sssilvar/HDD Data/Universidad/MSc/Thesis/Dataset/FreeSurferSD'
    dataset_folder = '/disk/Data/dataset/'
    # dataset_folder = '/disk/Data/center_simulation/center_2/input'
    csv_data = os.path.join(root, 'param', 'data_df.csv')
    regions = os.path.join(root, 'param', 'FreeSurferColorLUT.csv')

    # Curvelet params
    number_of_scales = 5
    number_of_angles = 4

    # Number of components for GMM
    n_components = [3, 5, 7]

    # Read datas into pandas DataFrame
    df = pd.read_csv(csv_data)
    df_regions = pd.read_csv(regions, index_col=['region_id'])

    for n_comp in n_components:
        feature_list = []

        # Start feature extraction
        for subject in df['folder']:
            print('\n\n[  INFO  ] Processing subject: ', subject)

            # Wipe feature dict
            features_subj = {}

            # Define filenames
            brainmask = os.path.join(dataset_folder, subject, 'mri', 'brainmask.mgz')
            aseg = os.path.join(dataset_folder, subject, 'mri', 'aseg.mgz')

            # Template files
            # mni_brainmask = os.path.join(root, 'param', 'fsaverage', 'brainmask.mgz')
            # mni_aseg = os.path.join(root, 'param', 'fsaverage', 'aseg.mgz')

            try:
                # Load images
                brainmask = nb.load(brainmask).get_data()
                aseg = nb.load(aseg).get_data()

                # mni_brainmask = nb.load(mni_brainmask).get_data()
                # mni_aseg = nb.load(mni_aseg).get_data()

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

                    if np.any(aseg == region) and region is not 0:
                        # Print region name
                        print('[  INFO  ] Processing ROI: ' + roi_name)

                        # Load subject's ROI
                        ix, iy, iz = np.where(aseg == region)
                        roi = np.array(brainmask[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)] * \
                                       (aseg == region)[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)])
                        print("\t - Subject's ROI shape: \t\t", roi.shape)

                        # Load template's ROI
                        # ix, iy, iz = np.where(mni_aseg == region)
                        # mni_roi = np.array(mni_aseg[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)])
                        # print("\t - Template's ROI shape: \t\t", mni_roi.shape)

                        # print(' ROI shape: ', np.shape(roi))
                        # plt.imshow(roi[:, :, int((max(iz) - min(iz)) / 2)])
                        # plt.show()

                        # Reshape it to template's
                        # roi = np.resize(roi, mni_roi.shape)
                        # print("\t - Subject's NEW ROI shape: \t", roi.shape)

                        # === CURVELET CALCULATION ===
                        A = ct.fdct3(roi.shape, nbs=number_of_scales, nba=number_of_angles, ac=True, norm=False,
                                     vec=True, cpx=False)

                        # Apply curvelet to the image
                        f = A.fwd(roi)

                        for scale in range(number_of_scales):
                            if scale == 0:
                                angles = [0]
                            elif scale == 1:
                                angles = range(0, number_of_angles)
                            elif scale % 2 == 0:
                                angles = range(0, int(scale * number_of_angles))
                            elif scale % 2 != 0:
                                angles = range(0, int((scale - 1) * number_of_angles))
                            else:
                                angles = []
                                raise ValueError('There is no angles inside the scale')

                            for angle in angles:
                                # Get index and generate a key name.
                                ix = A.index(scale, angle)
                                key = 'sca_%d_ang_%d' % (scale, angle)
                                try:
                                    # print('Getting scale %d | angle %d' % (scale, angle))
                                    data = np.ravel(f[ix[0]:ix[1]])
                                    # Fit GMM
                                    gmm = GaussianMixture(n_components=n_comp)
                                    gmm = gmm.fit(X=np.expand_dims(data, 1))

                                    for i in range(n_comp):
                                        features_subj[key + '_mean_' + roi_name + '_' + str(i)] = gmm.means_[i, 0]
                                        features_subj[key + '_cov_' + roi_name + '_' + str(i)] = gmm.covariances_[i, 0, 0]
                                except ValueError as e:
                                    print('[  ERROR  ] ', e, ' | ROI voxels: ', len(data), ' | ROI : ',
                                          roi_name)
                    else:
                        print('[  INFO  ] ROI not found')
                        # Save null results
                        for scale in range(number_of_scales):
                            if scale == 0:
                                angles = [0]
                            elif scale == 1:
                                angles = range(0, number_of_angles)
                            elif scale % 2 == 0:
                                angles = range(0, int(scale * number_of_angles))
                            elif scale % 2 != 0:
                                angles = range(0, int((scale - 1) * number_of_angles))
                            else:
                                angles = []
                                raise ValueError('There is no angles inside the scale')

                            for angle in angles:
                                # Save Null data
                                key = 'sca_%d_ang_%d' % (scale, angle)
                                for i in range(n_comp):
                                    features_subj[key + 'mean_' + roi_name + '_' + str(i)] = np.nan
                                    features_subj[key + 'cov_' + roi_name + '_' + str(i)] = np.nan
                feature_list.append(features_subj)
            except IOError as e:
                print('[  ERROR  ] ' + str(e))

        # Create a DataFrame
        df_features = pd.DataFrame(feature_list)
        df_features.to_csv(os.path.join(root, 'features', 'curvelet_gmm_features_%d_comp.csv' % n_comp))
