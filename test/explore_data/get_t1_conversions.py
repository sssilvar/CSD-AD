import os
from zipfile import ZipFile
from os.path import join, dirname, realpath

import numpy as np
import pandas as pd

# Set root folder: root
root = dirname(dirname(dirname(realpath(__file__))))


if __name__ == '__main__':
    os.system('clear')
    # ADNI zipped_folder
    adni_folder = '/run/media/ssilvari/Smith_2T_WD/Databases/ADNI_3T_3Y'
    destination_folder = '/disk/conversion/images'

    # Load MCIc and MCInc subjects data
    adni_prog = pd.read_csv(join(root, 'param/common/adnimerge_conversions_v2.csv'), index_col='PTID')
    adni_no_prog = pd.read_csv(join(root, 'param/common/adnimerge_MCInc_v2.csv'), index_col='PTID')
    df_mci = pd.concat([adni_no_prog, adni_prog], axis=0)

    # Load zip files
    adni_zips = ['ADNI1_Complete_3Yr_1.5T_%d.zip' % i for i in range(1,11)]
    subjects = {k: [] for k in adni_zips}
    for zip_file in adni_zips:
        print('[  INFO  ] Loading: %s' % zip_file)
        try:
            with ZipFile(join(adni_folder, zip_file), 'r') as z:
                subjects[zip_file].append(z.namelist())
        except Exception as e:
            print(e)
    
    # Create a DataFrame from list of files
    data = dict([ (k, pd.Series(v)) for k, v in subjects.items() ])
    subj_list = pd.DataFrame(data)

    for subject in df_mci.index:
        for col in subj_list.columns:
            try:
                files = map(str, subj_list.loc[0, col])
                if any(subject in f for f in files):
                    print('Subject %s found in %s' % (subject, col))
            except TypeError as e:
                # print(e)
                pass

