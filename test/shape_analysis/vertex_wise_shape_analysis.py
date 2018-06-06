import os

import numpy as np
from scipy.stats import ttest_ind

if __name__ == '__main__':
    # TEST
    data_a = np.random.normal(10, 10, 50)
    data_b = np.random.normal(18, 10, 55)

    # Start t-test
    t, p = ttest_ind(data_a, data_b)

    print(p)
