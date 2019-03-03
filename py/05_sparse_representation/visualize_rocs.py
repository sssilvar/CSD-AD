import os
from configparser import ConfigParser
from os.path import join, dirname, realpath

import pandas as pd
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load config
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))

    # Load params
    data_folder = cfg.get('dirs', 'sphere_mapping')
    roc_folder = join(data_folder, 'curvelets_non_split', 'ROC')
    n_folds = 5

    # Classifiers, image types and months
    classifiers = ['svm', 'rf']
    img_types = ['gradient', 'sobel']
    times = [24, 36, 60]

    for img_type in img_types:
        for clf in classifiers:
            clf_name = 'Random Forest' if clf == 'rf' else 'SVM'
            plt.figure(figsize=(7, 7))
            for t in times:
                # gradient_curvelet_features_non_split_aio_5_fold_svm_60_months_final.csv
                data_file = join(roc_folder,
                                 '{name}_curvelet_features_non_split_aio_{folds}_fold_{clf}_{time}_months_final.csv'
                                 .format(name=img_type, folds=n_folds, clf=clf, time=t))
                df = pd.read_csv(data_file)

                # Plot ROC
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(
                    df['mean_fpr'], df['mean_tpr'],
                    label='{time} (AUC = {auc:.2f})'.format(
                        time='{} Months'.format(t),
                        auc=df['mean_auc'][0]))
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.legend()

                plt.title('Mean ROC - {} - {}'.format(clf_name, img_type.capitalize()))
    plt.show()
