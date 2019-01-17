#!/bin/env python
import os
import re
import math
from zipfile import ZipFile
from os.path import join, dirname, realpath

import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))


def print_report(mcic, mcinc):
    print('====== ADNI MCI Report ======')
    print('\t- MCIc subjects: %d' % len(mcic))
    print('\t\t- ADNI1: %d' % len(mcic[mcic['COLPROT'] == 'ADNI1']))
    print('\t\t- ADNI2: %d' % len(mcic[mcic['COLPROT'] == 'ADNI2']))
    print('\t\t- ADNI3: %d' % len(mcic[mcic['COLPROT'] == 'ADNI3']))
    print('\t\t- ADNIGO: %d' % len(mcic[mcic['COLPROT'] == 'ADNIGO']))
    
    print('\t- MCInc subjects: %d' % len(mcinc))
    print('\t\t- ADNI1: %d' % len(mcinc[mcinc['COLPROT'] == 'ADNI1']))
    print('\t\t- ADNI2: %d' % len(mcinc[mcinc['COLPROT'] == 'ADNI2']))
    print('\t\t- ADNI3: %d' % len(mcinc[mcinc['COLPROT'] == 'ADNI3']))
    print('\t\t- ADNIGO: %d' % len(mcinc[mcinc['COLPROT'] == 'ADNIGO']))

    print('\t- Total subjects: %d' % (len(mcic) + len(mcinc)))


def get_files_from_study(adni_data):
    adni_files = []
    for adni_x, db_folder in adni_data.items():
        print('[  INFO  ] Analyzing zipped files for {}'.format(adni_x))
        adni_files += [join(db_folder, f) for f in os.listdir(db_folder) if '.zip' in f and 'metadata' not in f]
    return adni_files


if __name__ == "__main__":
    os.system('clear')
    # Load adnimerge files
    mcic_file = join(root, 'param', 'common', 'adnimerge_conversions_v2.csv')
    mcinc_file = join(root, 'param', 'common', 'adnimerge_MCInc_v2.csv')

    mcic = pd.read_csv(mcic_file, index_col='PTID', low_memory=False)
    mcinc = pd.read_csv(mcinc_file, index_col='PTID', low_memory=False)

    # Create a global dataframe: adni_df
    adni_df = pd.concat([mcic, mcinc], axis=0, ignore_index=False, sort=False)
    adni_df = adni_df[adni_df['COLPROT'] == 'ADNIGO']
    adni_df['IMAGEUID'] = adni_df['IMAGEUID'].fillna(8888888888).astype('int').astype('str')
    print(adni_df['IMAGEUID'].head())

    # Print a brief report
    print_report(mcic, mcinc)

    # Define studies and folders
    adni_data = {
        'ADNI1': '/run/media/ssilvari/Smith_2T_WD/Databases/ADNI/ADNI_1.5T_3Y',
        'ADNI2-3-GO': '/disk/data/ADNI/ADNI_2_GO'
    }

    adni_files = get_files_from_study(adni_data)
    print(adni_files)
    
    for zip_file in adni_files:
        # Extract files
        zf = ZipFile(zip_file, 'r')
        files_in_zip = zf.namelist()
        
        # print(files_in_zip)
        for subject in adni_df.index:
            uid = adni_df.loc[subject, 'IMAGEUID']
            
            # regex = 'ADNI/' + subject + '.*' + uid + '.*.nii'
            regex = 'ADNI/' + subject + '.*.nii'
            r = re.compile(regex)
            # print(regex)
            buff = list(filter(r.match, files_in_zip))
            if buff:
                if len(buff) == 1:
                    print('[  INFO  ] Extracting {}...'.format(subject))
                    try:
                        zf.extract(buff[0], '/disk/data/ADNI_SELECTED/')
                    except Exception as e:
                        print('[  ERROR  ] Error extracting the file. {}'.format(e))

