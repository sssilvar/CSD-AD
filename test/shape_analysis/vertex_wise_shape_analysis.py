import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

up = os.path.dirname
root = up(up(up(os.path.realpath(__file__))))

if __name__ == '__main__':
    csv_data = os.path.join(root, 'param', 'data_df.csv')
    csv_thickness = '/home/sssilvar/Documents/data/cortical_shape_analysis/groupfile_thick_2e-4.csv'

    # Load data
    df_dataset = pd.read_csv(csv_data)
    df_thickness = pd.read_csv(csv_thickness)

    # TEST
    data_a = np.random.normal(10, 10, 50)
    data_b = np.random.normal(18, 10, 55)

    # Start t-test
    t, p = ttest_ind(data_a, data_b)

    print(p)
