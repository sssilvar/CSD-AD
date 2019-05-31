#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/create_features_matrix.py"
NOTIFIER="${CURRENT_DIR}/../lib/email_notifier.py -subject 'Index calculation finished'"
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

            CMD=".${SCRIPT} -tk ${tk} -overlap ${overlap} -ns ${ns}"
            echo ${CMD}
        fi
    fi
done
