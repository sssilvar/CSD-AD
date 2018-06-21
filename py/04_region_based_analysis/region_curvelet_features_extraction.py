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


def get_n_angles(scale):
    if scale == 0:
        return [0]
    elif scale == 1:
        return range(0, number_of_angles)
    elif scale % 2 == 0:
        return range(0, int(scale * number_of_angles))
    elif scale % 2 != 0:
        return range(0, int((scale - 1) * number_of_angles))
    else:
        raise ValueError('There is no angles inside the scale')


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
    n_components = [3, 7, 11]

    # Read datas into pandas DataFrame
    df = pd.read_csv(csv_data)
    df_regions = pd.read_csv(regions, index_col=['region_id'])

    for n_comp in n_components:
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
                    mask = aseg == region

                    if np.any(mask) and region is not 0:
                        # Load subject's ROI
                        ix, iy, iz = np.where(mask)
                        roi = np.array(brainmask[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)] * \
                                        mask[min(ix): max(ix), min(iy): max(iy), min(iz): max(iz)])
                        print("\t - Subject's ROI shape: \t\t", roi.shape)
                        sx, sy, sz = roi.shape

                        if sx > 3 and sy > 3 and sz > 3:
                            # Print region name
                            print('[  INFO  ] Processing ROI: ' + roi_name)

                            # === CURVELET CALCULATION ===
                            A = ct.fdct3(roi.shape, nbs=number_of_scales, nba=number_of_angles, ac=True, norm=False,
                                        vec=True, cpx=False)

                            # Apply curvelet to the image
                            f = A.fwd(roi)

                            for scale in range(number_of_scales):
                                for angle in get_n_angles(scale):
                                    # Get index and generate a key name.
                                    ix = A.index(scale, angle)
                                    key = 'sca_%d_ang_%d' % (scale, angle)
                                    try:
                                        # print('Getting scale %d | angle %d' % (scale, angle))
                                        data = np.ravel(f[ix[0]:ix[1]])
                                        # Fit GMM
                                        gmm = GaussianMixture(n_components=n_comp, random_state=42)
                                        gmm = gmm.fit(X=np.expand_dims(data, 1))

                                        # Evaluate and visualize GMM
                                        gmm_x = np.linspace(0, 253, 256)
                                        gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

                                        # plt.hist(data, 255, [2, 256], normed=True, color='b')
                                        # plt.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
                                        # plt.title('ROI (' + key + '): ' + roi_name)
                                        # plt.show()

                                        for i in range(n_comp):
                                            features_subj[key + '_mean_' + roi_name + '_' + str(i)] = gmm.means_[i, 0]
                                            features_subj[key + '_cov_' + roi_name + '_' + str(i)] = gmm.covariances_[i, 0, 0]
                                    except ValueError as e:
                                        print('[  ERROR  ] ', e, ' | ROI voxels: ', len(data), ' | ROI : ', roi_name)
                                        for i in range(n_comp):
                                            features_subj[key + '_mean_' + roi_name + '_' + str(i)] = np.nan
                                            features_subj[key + '_cov_' + roi_name + '_' + str(i)] = np.nan
                        else:
                            print('[  WARNING  ] Shape is not enough')
                            """If ROI shape not enough"""
                            # Save null results
                            for scale in range(number_of_scales):
                                for angle in get_n_angles(scale):
                                    # Save Null data
                                    key = 'sca_%d_ang_%d' % (scale, angle)
                                    for i in range(n_comp):
                                        features_subj[key + '_mean_' + roi_name + '_' + str(i)] = np.nan
                                        features_subj[key + '_cov_' + roi_name + '_' + str(i)] = np.nan
                    else:
                        print('[  WARNING  ] ROI not found')
                        # Save null results
                        for scale in range(number_of_scales):
                            for angle in get_n_angles(scale):
                                # Save Null data
                                key = 'sca_%d_ang_%d' % (scale, angle)
                                for i in range(n_comp):
                                    features_subj[key + '_mean_' + roi_name + '_' + str(i)] = np.nan
                                    features_subj[key + '_cov_' + roi_name + '_' + str(i)] = np.nan

                # SAVE RESULTS
                subject_output_dir = os.path.join(root, 'features', 'curvelets', subject)

                try:
                    os.mkdir(subject_output_dir)
                except OSError:
                    pass

                np.save(os.path.join(subject_output_dir, 'curvelet_gmm_%d_comp' % n_comp), features_subj)

            except IOError as e:
                print('[  ERROR  ] ' + str(e))

        # # Create a DataFrame
        # print('[  INFO  ] Saving file...')
        # df_features = pd.DataFrame(feature_list)
        # df_features.to_csv(os.path.join(root, 'features', 'curvelet_gmm_features_%d_comp_part1.csv' % n_comp))
        # print('[  DONE!  ]')
