import os
import argparse
from os.path import dirname, realpath, join, basename

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

# Set root folder: root
root = dirname(dirname(dirname(realpath(__file__))))

# Set colors plot
sns.set(color_codes=True)

if __name__ == '__main__':
    # Clear screen
    os.system('clear')
    
    # Set the file and read it
    # data_csv = '/disk/conversion/features/curvelets/curvelet_gmm_3_comp.csv'
    data_csv = '/disk/Datasets/ADNI/screening_aseg/ADNI/wmh_voxels.csv'

    # df = pd.read_csv(data_csv, index_col='sid') # For gmm_X_comp.csv
    # df = pd.read_csv(data_csv, index_col='subject_id')
    df = pd.read_csv(data_csv, index_col=0)
    
    # Read database_targets
    for c in ['Unnamed: 0', 'target', 'target_name', 'index']:
            if c in df.columns:
                df = df.drop(c, axis=1)

    df = df.dropna(axis=1, how='any')
    print(df.head())
    cols = df.columns

    # Load progressions from MCI to Dementia (MCIc)
    adni_prog = pd.read_csv(join(root, 'param/common/adnimerge_conversions_v2.csv'), index_col='PTID')
    adni_no_prog = pd.read_csv(join(root, 'param/common/adnimerge_MCInc_v2.csv'), index_col='PTID')
    adnimerge = pd.read_csv(join(root, 'param/common/adnimerge.csv'), index_col='PTID', low_memory=False)
    adnimerge = adnimerge[adnimerge['VISCODE'] == 'bl']
    df_prog = pd.concat([adni_no_prog, adni_prog], axis=0)

    # Assign labels
    labels = list(np.empty(len(df.index)))
    for i, sid in enumerate(df.index):
        if sid in adni_prog.index:
            labels[i] = 'MCIc'
        elif sid in adni_no_prog.index:
            labels[i] = 'MCInc'
        elif sid in adnimerge.index:
            labels[i] = adnimerge.loc[sid, 'DX.bl']
        else:
            labels[i] = 'NA'

    df['label'] = labels
    # df = df[df['label'] != 'NA']
    # Perform Variability analysis
    X = df.drop('label', axis=1)
    
    # Normalize w.r.t. Intracranial Volume (ICV)
    icv = df_prog.loc[df.index, 'ICV']
    icv = icv.fillna(icv.mean())
    X = X.divide(icv, axis=0)

    # Standardize data
    X = StandardScaler().fit_transform(X)

    # Load and preprocess commond data
    Y = df_prog.loc[df.index, ['PTGENDER', 'AGE']] #, 'PTRACCAT', 'SITE']#, 'APOE4', 'FDG']
    Y['AGE2'] = Y['AGE'] ** 2
    # Y['SITE'] = Y['SITE'].astype('category').cat.codes
    # Y['PTRACCAT'] = Y['PTRACCAT'].astype('category').cat.codes
    Y['PTGENDER'] = Y['PTGENDER'].astype('category').cat.codes
    Y = Y.fillna(Y.mean())
    print(Y.shape)
    print(X.shape)
    print(icv.shape)

    W = np.linalg.inv(Y.T.dot(Y)).dot(Y.T.dot(X))

    # Substract effect of age
    X = X - Y.dot(W)
    
    # Save_corrected_data: cdf
    cdf = X.copy()
    cdf.columns = cols
    cdf['label'] = df['label']
    corrected_csv = join(data_csv[:-4] + '_corrected.csv')
    cdf.to_csv(corrected_csv)
    print('CSV saved!')
    
    # Reduce dimmensionality: X_ld
    n_comp = 5 if X.shape[1] > 5 else X.shape[1]
    X_ld = PCA(n_components=n_comp).fit_transform(X)
    
    #Convert in DataFrame
    cols = ['PC%d' % (c + 1) for c in range(X_ld.shape[1])]
    X_ld = pd.DataFrame(data=X_ld, columns=cols, index=df.index)
    X_ld['label'] = df['label']
    X_ld = X_ld[X_ld['label'] != 'NA']
    print(X_ld.shape)
    print(X_ld['label'].value_counts())

    # Plot results
    palette = 'Set1'
    for ca in cols[:-1]:
        for cb in cols[1:]:
            if ca is not cb:
                sns.lmplot( x=ca,
                            y=cb,
                            data=X_ld,
                            hue='label',
                            fit_reg=False,
                            palette=palette)
                
                results_folder = join('/tmp/', basename(data_csv)[:-4])
                os.system('mkdir ' + results_folder)
                plt.savefig(join(results_folder, '%s_vs_%s.png' % (ca, cb)), bbox_inches='tight', dpi=300)

    X_tsne = TSNE(learning_rate=100, random_state=42).fit_transform(X_ld.drop('label', axis=1))
    X_ld['TSNE_1'] = X_tsne[:, 0]
    X_ld['TSNE_2'] = X_tsne[:, 1]
    
    sns.lmplot( x='TSNE_1',
                y='TSNE_2',
                data=X_ld,
                hue='label',
                fit_reg=False,
                palette=palette)
    
    plt.savefig(join(results_folder, 'tsne_2D.png'), bbox_inches='tight', dpi=300)



    

