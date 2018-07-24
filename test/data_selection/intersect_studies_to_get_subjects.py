import os
from os.path import join, dirname, realpath

import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    cf = dirname(realpath(__file__))
    data_folder = join(cf, 'data')
    files = os.listdir(data_folder)
    
    for i, f in enumerate(files):
        f = os.path.join(data_folder, f)

        if i == 0:
            df = pd.read_csv(f)
        else:
            df = df.append(pd.read_csv(f))

    df = df.drop_duplicates(subset=['sid', 'visit'], keep='last')

    # Load ADNI data to get UID
    # adnimerge_file = join(root, 'param', 'common', 'adnimerge.csv')
    adnimerge_file = '/home/ssilvari/Downloads/ADNIMERGE.csv'
    adnimerge_df = pd.read_csv(adnimerge_file, index_col=0)

    adnimerge_df['key'] = adnimerge_df['PTID'] + adnimerge_df['VISCODE']
    keys = df['sid'] + df['visit'].str.lower().replace('m00', 'bl')

    adnimerge_df = adnimerge_df.set_index('key')
    conversions_df = adnimerge_df.reindex(keys)

    print(df.info())
    print(conversions_df.info())
    conversions_df.to_csv(join(root, 'param', 'common', 'conversion_adnimerge.csv'))
    
    