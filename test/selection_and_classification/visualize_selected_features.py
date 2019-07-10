#!/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

if __name__ == '__main__':
    selected_feats_file = '/home/ssilvari/Downloads/results/results_norm_mrmr/selected_features_tk_25_overlap_0.csv'
    df = pd.read_csv(selected_feats_file, index_col=0)
    print(df.head())

    top_feats = df['feature'].value_counts().head(20)
    df_top = pd.DataFrame()
    for feat_name in top_feats.index:
        df_top = df_top.append(df.query(f'feature == "{feat_name}"'))

    print(df_top.head())
    df_top.sort_values(by=['feature'], inplace=True)
    g = sns.catplot(y='feature', col='time', data=df_top, kind='count')
    g.set_xticklabels(rotation=90)
    plt.show()
