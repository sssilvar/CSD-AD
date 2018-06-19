#!/usr/bin/env bash

# Get current dir
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT_FOLDER="/user/ssilvari/home/Documents/structural/CSD-AD"
OUTPUT_FOLDER=${SCRIPT_FOLDER}"/features"
DATASET_FOLDER="/disk/Data/dataset"
PROXY=""

CMD="bash "${CURRENT_DIR}"/build.sh "${DATASET_FOLDER}" "${OUTPUT_FOLDER}" "${SCRIPT_FOLDER}" "${PROXY}
echo ${CMD}
eval ${CMD}

eval "chmod -R 766 /output "