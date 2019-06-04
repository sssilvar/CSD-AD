#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/classify_all.sh"
ROC_SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/visualize_rocs.py"
NOTIFIER="${CURRENT_DIR}/../lib/telegram_notifier.py"
MAPPED_FOLDER="/home/jullygh/sssilvar/Documents/Dataset/mapped"
N_SUBJ=829

FOLDERS=($(find ${MAPPED_FOLDER} -type d -name "ADNI*tk*overlap*"))

for folder in ${FOLDERS[@]} ; do
    curv_folder="${folder}/curvelet/"
    count_files=`ls -1 ${curv_folder}*.csv 2>/dev/null | wc -l`
    if [[ ${count_files} > 0 ]]; then
        n=$(ls ${folder} | grep _S_ | wc -l)
        if [[ "${n}" == "${N_SUBJ}" ]]; then
            echo "Processing ${folder} (${n} subjects)"
            tk=$(echo basename ${folder} | awk -F'_' '{ print $5 }')
            overlap=$(echo basename ${folder} | awk -F'_' '{ print $7 }')
            ns=$(echo basename ${folder} | awk -F'_' '{ print $9 }')

            eval "rm -rf ${curv_folder}/ROC"
#            CMD="${SCRIPT} ${curv_folder}"
            CMD="ls"
            CMD="${CMD} && ${ROC_SCRIPT} ${curv_folder}"
            CMD="${CMD} && ${NOTIFIER} -msg 'Classification for tk=${tk}, overlap=${overlap} and ns=${ns} done.'"
            CMD="tmux new-session -d -s \"classification_tk_${tk}_ov_${overlap}_${ns}\" \"${CMD}\""

            echo ${CMD}
            eval ${CMD}
        fi
    fi
done
