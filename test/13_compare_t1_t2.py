import os
import argparse
from os.path import dirname, realpath, join, basename

import numpy as np
import pandas as pd

# Set root folder: root
root = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':
    # Load results_file
    res_file = '/run/media/ssilvari/Smith_2T_WD/Databases/ADNI_T2_FLAIR/T2_screening.csv'
    df_res = pd.read_csv(res_file, index_col='Subject')
    
    # # Load progressions from MCI to Dementia (MCIc)
    # adni_prog = pd.read_csv(join(root, 'param/common/adnimerge_conversions_v2.csv'), index_col='PTID')
    # adni_no_prog = pd.read_csv(join(root, 'param/common/adnimerge_MCInc_v2.csv'), index_col='PTID')
    # df_prog = pd.concat([adni_no_prog, adni_prog], axis=0)
    
    # Using ADNIMERGE
    df_prog = adni_prog = pd.read_csv(join(root, 'param/common/adnimerge.csv'), index_col='PTID', low_memory=False)
    df_prog = df_prog[df_prog['DX.bl'] != 'SMC']

    df_coll = pd.read_csv('/user/ssilvari/home/Downloads/idaSearch_10_18_2018.csv', index_col='Subject ID')

    t1_set = set(df_res.index)
    t2_set = set(df_prog.index)

    intersection = t1_set.intersection(t2_set)
    diff = t2_set.difference(df_coll)
    print('Progession Data shape: %s' % str(df_prog.shape))
    print('T2 Data shape: %s' % str(df_res.shape))
    print(list(diff)[:10])
    print(len(intersection))
