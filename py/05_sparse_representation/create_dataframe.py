import os
from glob import glob
from os.path import join, dirname, realpath, basename

import numpy as np
import pandas as pd

root = dirname(dirname(dirname(realpath(__file__))))  # root folder


if __name__ == '__main__':
    out_folder = join(root, 'output')
    scales = [5, 6, 7, 9]
    angles = [8, 16, 32]
    img_type = 'gradient'  # Or 'intensity

    for scale in scales:
        print('\n[  OK  ] Processing scale %d' % scale)
        for angle in angles:
            folder = join(out_folder, 'curv_feats_%s_nscales_%d_nangles_%d' % (img_type, scale, angle))
            if os.path.exists(folder):
                # Get *npy files
                npz_files = [npf for npf in glob(join(folder, '*.npz'))]
                
                # === Create DataFrame from all the NPZ present in the folder ====
                for i, npz_file in enumerate(npz_files):
                    subject = dict(np.load(npz_file))
                    subject_id = basename(npz_file[:-4])
                    subject_series = pd.Series(subject, name=subject_id)
                    
                    # Initialize dataframe if necessary
                    print('[  INFO  ] Adding subject: %s' % subject_id)
                    if i == 0:
                        df = subject_series.to_frame().transpose()
                    else:
                        df = df.append(subject_series)
            else:
                print('[  ERROR  ] Folder not found')