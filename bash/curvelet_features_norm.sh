#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/create_features_matrix_norm.py"
NOTIFIER="${CURRENT_DIR}/../lib/telegram_notifier.py"
MAPPED_FOLDER="/home/jullygh/sssilvar/Documents/Dataset/mapped"
N_SUBJ=829

FOLDERS=($(find ${MAPPED_FOLDER} -type d -name "ADNI*tk*overlap*"))

for folder in ${FOLDERS[@]} ; do
    if [[ ! -d "${folder}/curvelet" ]]; then
        n=$(ls ${folder} | grep _S_ | wc -l)
        if [[ "${n}" == "${N_SUBJ}" ]]; then
            echo "Processing ${folder} (${n} subjects)"
            tk=$(echo basename ${folder} | awk -F'_' '{ print $5 }')
            overlap=$(echo basename ${folder} | awk -F'_' '{ print $7 }')
            ns=$(echo basename ${folder} | awk -F'_' '{ print $9 }')

            CMD="${SCRIPT} -tk ${tk} -overlap ${overlap} -ns ${ns}"
            CMD="${CMD} && ${NOTIFIER} -msg 'Curvelet calculation for tk=${tk} and overlap=${overlap} done.'"
            CMD="tmux new-session -d -s \"curvelet_tk_${tk}_ov_${overlap}\" \"${CMD}\""

            echo ${CMD}
            eval ${CMD}
        fi
    fi
done
