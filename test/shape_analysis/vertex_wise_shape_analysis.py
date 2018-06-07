import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

plt.style.use('ggplot')

if __name__ == '__main__':
    csv_data = os.path.join(root, 'param', 'data_df.csv')
    csv_features = '/home/sssilvar/Documents/data/cortical_shape_analysis/groupfile_thick_2e-4.csv'
    # csv_features = '/home/sssilvar/Documents/data/cortical_shape_analysis/output/groupfile_thick.csv'

    # Load data
    df_dataset = pd.read_csv(csv_data)
    df_thickness = pd.read_csv(csv_features, index_col=0)

    mci_c_subjects = df_dataset.loc[df_dataset['target_categories'] == 'MCIc']['folder']
    mci_nc_subjects = df_dataset.loc[df_dataset['target_categories'] == 'MCInc']['folder']

    df_mci_c = df_thickness.loc[df_thickness.index.isin(mci_c_subjects)]
    df_mci_nc = df_thickness.loc[df_thickness.index.isin(mci_nc_subjects)]

    t_stat = []
    p_val = []

    for column in df_mci_c:
        t, p = ttest_ind(df_mci_nc[column], df_mci_c[column])

        t_stat.append(t)
        p_val.append(p)

    print('[  INFO  ] Saving results in ./output')
    np.array(t_stat).tofile(os.path.join(root, 'output', 't_stat_' + os.path.basename(csv_features[:-4]) + '.raw'))
    np.array(p_val).tofile(os.path.join(root, 'output', 'p_value_' + os.path.basename(csv_features[:-4]) + '.raw'))

    print('DONE!')

    # f, ar = plt.subplots(2, 1, sharex=True)
    # ar[0].plot(p_val)
    # ar[0].plot(np.ones_like(p_val) * 0.05, 'b--')
    # ar[0].legend(['p-value', '0.05'])
    # ar[0].set_yscale('log')
    #
    # ar[1].plot(t_stat)
    # ar[1].legend(['t-statistic'])
    #
    # plt.show()
