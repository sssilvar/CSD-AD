#!/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_context('paper')
sns.set(font_scale=1.2)

if __name__ == '__main__':
    selected_feats_file = '/home/ssilvari/Downloads/results/results_angles_mrmr/selected_features_tk_25_overlap_0.csv'
    df = pd.read_csv(selected_feats_file, index_col=0)
    print(df.head())

    df['band'] = df['feature'].map(lambda x: 'Scale {} Sub {}'.format(x.split('_')[0], x.split('_')[1]))

    top_feats = df['band'].value_counts()
    top_feats = top_feats[top_feats > 10].head(20)

    df_top = pd.DataFrame()
    for feat_name in top_feats.index:
        df_top = df_top.append(df.query(f'band == "{feat_name}"'))

    print(df_top.head())
    # df_top.sort_values(by=['feature'], inplace=True)
    g = sns.catplot(y='band', col='time', data=df_top, kind='count')
    g.set_xticklabels(rotation=90)
    g.set_xlabels('Counts', fontsize=21)
    g.set_ylabels('Curvelet Sub-band', fontsize=21)
    plt.show()
