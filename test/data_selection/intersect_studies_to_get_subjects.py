import os

import pandas as pd

if __name__ == '__main__':
    cf = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(cf, 'data')
    files = os.listdir(data_folder)
    
    for i, f in enumerate(files):
        f = os.path.join(data_folder, f)

        if i == 0:
            df = pd.read_csv(f)
        else:
            df = df.append(pd.read_csv(f))

df.drop_duplicates(subset=['sid', 'visit'], keep=False)
print(df.info())