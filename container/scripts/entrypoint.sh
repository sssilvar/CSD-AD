#!/bin/bash
echo -e "\n\n==== CURVELET FEATURE EXTRACTION ===\n"

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# eval "python2 "${CURRENT_DIR}"/test.py"
eval "python2 /py/py/05_sparse_representation/feature_extraction.py -f "${DATA_FOLDER}
eval "chmod -R 766 "${DATA_FOLDER}