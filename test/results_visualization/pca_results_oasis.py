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
    data_csv = '/disk/Datasets/OASIS/oasis_extracted/wmh_voxels.csv'
    df = pd.read_csv(data_csv, index_col=0)
    
    # Read database_targets
    for c in ['Unnamed: 0', 'target', 'target_name', 'index']:
            if c in df.columns:
                df = df.drop(c, axis=1)

    df = df.dropna(axis=1, how='any')
    cols = df.columns

    # Load subjects' data
    df_data = pd.read_csv('/disk/Datasets/OASIS/info/oasis_cross-sectional.csv', index_col='ID')
    df_vols = pd.read_csv('/disk/Datasets/OASIS/info/oasis_cross-sectional-reliability.csv', index_col='ID')
    df_data['CDR'] = df_data['CDR'].fillna(0)

    print(df.head())
    print(df_data.head())
    print(df_vols.head())

    # Assign labels
    labels = list(np.empty(len(df.index)))
    for i, sid in enumerate(df.index):
        cdr = df_data.loc[sid, 'CDR']
        if cdr < 0.5 :
            labels[i] = 'HC'
        elif cdr >= 0.5 and cdr < 1:
            labels[i] = 'mild-AD'
        elif cdr >= 1 and cdr <= 2:
            labels[i] = 'moderate-AD'
        elif cdr > 2:
            labels[i] = 'severe-AD'

    df['label'] = labels
    X = df.drop('label', axis=1)
    
    # Normalize w.r.t. Intracranial Volume (eTIV)
    icv = df_vols.loc[df.index, 'eTIV']
    icv = icv.fillna(icv.mean())
    X = X.divide(icv, axis=0)

    # Standardize data
    X = StandardScaler().fit_transform(X)

    # Load and preprocess commond data
    Y = df_data.loc[df.index, ['M/F', 'Age']]
    Y['Age2'] = Y['Age'] ** 2
    Y['M/F'] = Y['M/F'].astype('category').cat.codes
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
