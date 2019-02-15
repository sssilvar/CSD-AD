from os.path import join, dirname, realpath

import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))


if __name__ == '__main__':
    csv_file = join(root, 'param', 'df_conversions_with_times.csv')
    print(csv_file)
    df = pd.read_csv(csv_file, index_col='PTID')
    print(df.head())

    for month in [0, 24, 36, 60]:
        print('For {} months:'.format(month))
        print(df.loc[(df['Month.STABLE'] >= month) | (df['Month.CONVERSION'] <= month), 'target'].value_counts())

    # Look for missing data
    groupfile = '/home/ssilvari/Documents/temp/ADNI_test/ADNI_FS_sphere_mapped/groupfile.csv'
    df_existent = pd.read_csv(groupfile, index_col=0)
    print('\n\nGroupfile existing Data:')
    print(df.target.value_counts())
    intersection_subjects = set(df.index) & set(df_existent.index)
    missed_subject = set(df.index) - set(df_existent.index)

    print('\n\nData report:')
    print('\t- No. Subjects in ADNIMERGE: %d' % len(intersection_subjects))
    print('\t- No. Subjects Missed: %d' % len(missed_subject))
    print('\t- Subjects missed:')
    for s in missed_subject:
        print(s)
