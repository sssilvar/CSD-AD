from os.path import join, realpath, dirname

import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load conversions file
    conversions_file = join(root, 'param/df_conversions_with_times.csv')
    df = pd.read_csv(conversions_file, index_col='PTID')
    print('\nGlobal stats:')
    print(df['target'].value_counts())
    print()

    # Set a conversion times of interes: toi
    toi = [24, 36, 60]
    for t in toi:
        df_mci = df.loc[(df['Month.CONVERSION'] <= t) | (df['Month.STABLE'] >= t)]
        print('Data distribution for {} months'.format(t))
        print(df_mci['target'].value_counts())
