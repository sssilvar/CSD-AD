import os
from configparser import ConfigParser
from os.path import join, dirname, realpath, isdir

import pandas as pd
import matplotlib.pyplot as plt

root = dirname(dirname(dirname(realpath(__file__))))
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load config
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    fmt = 'png'  # Alt. 'eps'

    # Load params
    data_folder = cfg.get('dirs', 'sphere_mapping')
    roc_folder = join(data_folder, 'curvelet', 'ROC')
    n_folds = 10

    # Classifiers, image types and months
    classifiers = ['svm', 'rf']
    img_types = ['sobel']
    times = [24, 36, 60]

    for img_type in img_types:
        for clf in classifiers:
            clf_name = 'Random Forest' if clf == 'rf' else 'SVM'
            plt.figure(figsize=(7, 7))
            for t in times:
                # gradient_curvelet_features_non_split_aio_5_fold_svm_60_months_final.csv
                file_pattern = '{name}_curvelet_features_non_split_aio_{folds}_fold_{clf}_{time}_months_final.csv'

                # # sobel_curvelet_features_4_scales_32_angles_aio_5_fold_rf_24_months_final.csv
                # file_pattern = '{name}_curvelet_features_4_scales_32_angles_aio_{folds}_fold_{clf}_{time}_months_final.csv'
                file_pattern = file_pattern.format(name=img_type, folds=n_folds, clf=clf, time=t)

                data_file = join(roc_folder, file_pattern)

                df = pd.read_csv(data_file)

                # Plot ROC
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(
                    df['mean_fpr'], df['mean_tpr'],
                    label='{time} (AUC = {auc:.2f})'.format(
                        time='{} Months'.format(t),
                        auc=df['mean_auc'][0]),
                    linewidth=2.5,
                    alpha=0.8
                )
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.legend()

                plt.title('Mean ROC - {} - {} - {} folds'.format(clf_name, img_type.capitalize(), n_folds))
            # Save figures
            fig_folder = join(roc_folder, 'roc_by_month')
            if not isdir(fig_folder):
                os.mkdir(fig_folder)

            fig_file = join(
                fig_folder,
                '{img_type}_{clf}_{folds}_folds'.format(
                    img_type=img_type,
                    clf=clf_name.lower().replace(' ', '_'),
                    folds=n_folds
                )
            )
            print('Saving figure at: {}'.format(fig_file))
            plt.savefig(fig_file + '.{}'.format(fmt), bbox_inches='tight')
