import os
import json
import pandas as pd
from multiprocessing import Pool

# Define params
root = os.path.join(os.getcwd(), '..', '..')
params_file = os.path.join(root, 'param', 'params.json')


# Started: 3:28 - Ended:
# mri_cvs_register --mov ubject_id --mni --openmp 8
def register_subject(subject_id):
    print('[ PROCESSING ] Subject: ' + subject_id)
    cmd = 'mri_csv_register --mov ' + subject_id + ' --mni --openmp 8'
    # print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # Load params
    with open(params_file) as json_file:
        jf = json.load(json_file)
        dataset_folder = jf['dataset_folder']
        data_file = jf['data_file']

    # Load dataset data into a DataFrame: df
    df = pd.read_csv(os.path.join(root, data_file))
    df = df.sort_values('folder')

    # Set SUBJECTS_DIR environment variable
    os.system('export SUBJECTS_DIR=' + dataset_folder)

    # Start multicore
    print('[  OK  ] Starting pool')
    pool = Pool(4)
    pool.map(register_subject, df['folder'])
    pool.close()
