#!/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_context('paper')
sns.set(font_scale=1.2)

if __name__ == '__main__':
    selected_feats_file = '/home/ssilvari/Documents/temp/ADNI_temp/mapped/ADNI_FS_mapped_tk_25_overlap_4_ns_1' \
                          '/selected_features_tk_25_overlap_4.csv'
    df = pd.read_csv(selected_feats_file, index_col=0)

    df['band'] = df['feature'].map(lambda x: 'Scale {} Sub {}'.format(x.split('_')[0], x.split('_')[1]))
    df['sphere'] = df['feature'].map(lambda x: 'Sphere {} to {}'.format(x.split('_')[3], x.split('_')[5]))

    print(df.head())

    # ========================
    # Get the top sub-bands
    # ========================
    top_feats = df['band'].value_counts()
    top_feats = top_feats[top_feats > 10].head(20)

    df_top_sub_bands = pd.DataFrame()
    for feat_name in top_feats.index:
        df_top_sub_bands = df_top_sub_bands.append(df.query(f'band == "{feat_name}"'))

    # df_top.sort_values(by=['feature'], inplace=True)
    # g = sns.catplot(y='band', col='time', data=df_top_sub_bands, kind='count')
    # g.set_xticklabels(rotation=90)
    # g.set_xlabels('Counts', fontsize=21)
    # g.set_ylabels('Curvelet Sub-band', fontsize=21)

    # ========================
    # Get the top spheres
    # ========================
    top_spheres = df['sphere'].value_counts()
    top_spheres = top_spheres[top_spheres > 10].head(20)
    df_top_sub_spheres = pd.DataFrame()
    for sphere_name in top_spheres.index:
        df_top_sub_spheres = df_top_sub_spheres.append(df.query(f'sphere == "{sphere_name}"'))
    print(df_top_sub_spheres.head())

    g = sns.catplot(y='sphere', col='time', data=df_top_sub_spheres, kind='count')
    g.set_xticklabels(rotation=90)
    g.set_xlabels('Counts', fontsize=21)
    g.set_ylabels('Sphere', fontsize=21)

    plt.show()
