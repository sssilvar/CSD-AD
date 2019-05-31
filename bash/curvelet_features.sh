#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/create_features_matrix.py"
NOTIFIER="${CURRENT_DIR}/../lib/email_notifier.py -subject 'Index calculation finished'"
MAPPED_FOLDER="/home/jullygh/sssilvar/Documents/Dataset/mapped"

TKS=(15 25 30)
OVERLAPS=(0 4 5 7 9)

for tk in ${TKS[@]}; do
    for overlap in ${OVERLAPS[@]} ; do
        FOLDER="${MAPPED_FOLDER}/ADNI_FS_mapped_tk_${tk}_overlap_${overlap}_ns_1/curvelet"
        if ![[ -d ${FOLDER} ]]; then
            echo "Processing folder ${FOLDER}"
        fi        
    done
done