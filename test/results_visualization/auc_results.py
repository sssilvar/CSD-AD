import os
from os.path import join, dirname, realpath

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sns.set(font_scale=2)
root = dirname(dirname(dirname(realpath(__file__))))

if __name__ == '__main__':
    # Load results file
    res_file = join(root, 'output/auc_results.csv')

    # Load CSV
    df = pd.read_csv(res_file)
    df['OVERLAP'] = df['OVERLAP'].astype('category')
    df['CLF'] = df['CLF'].astype('category')

    times = [24, 36, 60]
    fig, ax = plt.subplots(ncols=3, figsize=(20, 7), sharey=True)
    for i, t in enumerate(times):
        sns.swarmplot(
            x="CLF", y=f"AUC_{t}M",
            hue="OVERLAP", data=df,
            s=20,
            ax=ax[i],
        )
        ax[0].set_ylabel('AUC')
        ax[i].set_title(f'{t} Months', fontsize=36)
    plt.show()
