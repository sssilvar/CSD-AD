#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/classify_all.sh"
ROC_SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/visualize_rocs.py"
NOTIFIER="${CURRENT_DIR}/../lib/telegram_notifier.py"
MAPPED_FOLDER="/home/jullygh/sssilvar/Documents/Dataset/mapped"
N_SUBJ=829

FOLDERS=($(find ${MAPPED_FOLDER} -type d -name "ADNI*tk*overlap*"))
eval "rm ~/Downloads/ROCS.tar.gz"

for folder in ${FOLDERS[@]} ; do
    curv_folder="${folder}/curvelet/"
    count_files=`ls -1 ${curv_folder}*.csv 2>/dev/null | wc -l`
    if [[ ${count_files} > 0 ]]; then
        n=$(ls ${folder} | grep _S_ | wc -l)
        if [[ "${n}" == "${N_SUBJ}" ]]; then
            if [[ -d "${curv_folder}/ROC" ]]; then
                CMD="tar -rvf ~/Downloads/ROCS.tar.gz ${curv_folder}/ROC"
                echo ${CMD}
                eval ${CMD}
            fi
        fi
    fi
done
