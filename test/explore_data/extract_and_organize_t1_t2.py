import os
from os.path import join
from zipfile import ZipFile

import pandas as pd

def get_list_of_files_in_zip(zip_files):
    """
    Recieves a list of zip files and
    returns the contents
    """
    files = {k: [] for k in zip_files}
    for zip_file in zip_files:
        print('[  INFO  ] Loading: %s' % zip_file)
        try:
            with ZipFile(join(zip_file), 'r') as z:
                files[zip_file] = files[zip_file] + z.namelist()
        except Exception as e:
            print(e)
    return files


if __name__ == '__main__':
    os.system('clear')
    # Set files containing T1 and T2 zipped images
    t1_zip_folder = '/run/media/ssilvari/Smith_2T_WD/Databases/ADNI_3T_3Y'
    t2_zip_folder = '/run/media/ssilvari/Smith_2T_WD/Databases/ADNI_T2_FLAIR'

    # Define output folder
    t2_df = pd.read_csv(join(t2_zip_folder, 'T2_screening.csv'), index_col='Subject')
    print(t2_df.head())

    # TODO: Fields: IMAGEUID
    # Define ADNI ZIP files
    adni_zips = ['/disk/Datasets/ADNI/T1_BRAINMASK/brainmask_baseline.zip'] + [join(t2_zip_folder, 'T2_screening_%d.zip' % i) for i in range(1,11)] # [join(t1_zip_folder, 'ADNI1_Complete_3Yr_1.5T_%d.zip' % i) for i in range(1,11)] + \

    # Get all the files inside ZIPs
    files = get_list_of_files_in_zip(adni_zips)

    # Start magic:
    t1_and_t2 = []
    for subject in t2_df.index:
        find = {k:[] for k in files.keys()}
        for key, val in files.items():
            for v in val:
                if subject in v:
                    print('%s Found in %s' % (subject, v))
                    find[key].append(v)
        print('\n')
        
        for key, val in find.items():
            if len(val) >= 2:
                t1_and_t2.append(1)

    print('Total subjects with T1 and t2 * %d ' % len(t1_and_t2))
        
    


    