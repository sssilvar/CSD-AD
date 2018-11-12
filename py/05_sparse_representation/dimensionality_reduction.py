#!/bin/env python
import os
import sys
from os.path import dirname, realpath, join, basename

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

# Set root folder: root
root = dirname(dirname(dirname(realpath(__file__))))
sns.set(color_codes=True)


if __name__ == '__main__':
    # os.system('clear')
    # Load Dataframe
    # data_file = '/home/ssilvari/Documents/results/sphere_mapped_curvelet_features/curv_feats_gradient_nscales_5_nangles_8/curv_feats_gradient_nscales_5_nangles_8.h5'
    data_file = sys.argv[1]
    print('[  INFO  ] Loading file (%s)...' % data_file)
    if data_file.endswith('.h5'):
        df = pd.read_hdf(data_file, key='features', mode='r', index_col=0, low_memory=False)
    else:
        df = pd.read_csv(data_file, index_col=0)
    
    # Remove useless columns
    try:
        df = df.drop(['target', 'n_scales', 'n_angles', 'subject'], axis=1)
    except Exception as e:
        pass
    df.fillna(0)
    print(df.head())
    
    # Load progressions from MCI to Dementia (MCIc)
    print('[  INFO  ] Loading confounders...')
    adni_prog = pd.read_csv(join(root, 'param/common/adnimerge_conversions_v2.csv'), index_col='PTID')
    adni_no_prog = pd.read_csv(join(root, 'param/common/adnimerge_MCInc_v2.csv'), index_col='PTID')
    adnimerge = pd.read_csv(join(root, 'param/common/adnimerge.csv'), index_col='PTID', low_memory=False)
    adnimerge = adnimerge[adnimerge['VISCODE'] == 'bl']
    df_prog = pd.concat([adni_no_prog, adni_prog], axis=0)

    Y = df_prog.loc[df.index, ['PTGENDER', 'AGE']] #, 'PTRACCAT', 'SITE']#, 'APOE4', 'FDG']
    Y['PTGENDER'] = Y['PTGENDER'].astype('category').cat.codes
    Y['AGE2'] = Y['AGE'] ** 2
    Y = Y.fillna(Y.mean())
    
    # Load Intra Cranial Volume (ICV)
    icv = df_prog.loc[df.index, 'ICV']
    icv = icv.fillna(icv.mean())
    
    # Normalize volumes by ICV (if file contains volumes)
    if 'vol' in data_file or 'wmh' in data_file:
        print('[  INFO  ] Normalizing by ICV...')
        df = df.divide(icv, axis=0)
        print(df.head())
    
    # Create a numpy array
    X = df.values

    # Substract effect of confounders
    print('[  INFO  ] Substracting confounders effect...')
    W = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(X))    
    X = X - Y.dot(W)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=20, whiten=True))
    ])

    # Perform the pipeline
    print('[  INFO  ] Reducing dimensionality (PCA)...')
    x_pca = pipeline.fit_transform(X)
    cols = ['PC%d' % c for c in range(1, x_pca.shape[1] + 1)]

    df_pca = pd.DataFrame(data=x_pca, columns=cols, index=df.index)
    print('[  INFO  ] PCA Resutls: \n %s' % df_pca.head())

    # Assign labels
    labels = list(np.empty(len(df_pca.index)))
    for i, sid in enumerate(df.index):
        if sid in adni_prog.index:
            labels[i] = 'MCIc'
        elif sid in adni_no_prog.index:
            labels[i] = 'MCInc'
        elif sid in adnimerge.index:
            labels[i] = adnimerge.loc[sid, 'DX.bl']
        else:
            labels[i] = 'NA'
    df_pca['label'] = labels

    # Filter data to plot
    df_pca = df_pca[(df_pca['label'] == 'MCIc') | (df_pca['label'] == 'MCInc')]
    
    # Plot results
    print('[  INFO  ] Plotting results...')
    palette = 'Set1'
    for ca in cols[:5]:
        for cb in cols[1:5]:
            if ca is not cb:
                sns.lmplot( x=ca,
                            y=cb,
                            data=df_pca,
                            hue='label',
                            fit_reg=False,
                            palette=palette)
                
                results_folder = join('/tmp/', basename(data_file)[:-3])
                os.system('mkdir ' + results_folder)
                plt.savefig(join(results_folder, '%s_vs_%s.png' % (ca, cb)), bbox_inches='tight', dpi=300)
    df_pca.to_csv(join(results_folder, basename(data_file)[:-3] + '_pca_20_comp.csv'))

    
