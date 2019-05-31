#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/create_features_matrix.py"
NOTIFIER="${CURRENT_DIR}/../lib/email_notifier.py -subject 'Index calculation finished'"
MAPPED_FOLDER="/home/jullygh/sssilvar/Documents/Dataset/mapped"

FOLDERS=($(ls ${MAPPED_FOLDER} | grep overlap))

for folder in ${FOLDERS[@]} ; do
    echo $folder
done
