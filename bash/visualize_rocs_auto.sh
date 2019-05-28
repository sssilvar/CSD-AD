#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/05_sparse_representation/visualize_rocs.py"
MAPPED_FOLDER="/home/ssilvari/Documents/temp/ADNI_temp/mapped"

# Define parameters
TKS=(25)
OVERLAPS=(0 4 5 9)
NS=(1 2)

for tk in ${TKS[@]} ; do
    for overlap in ${OVERLAPS[@]} ; do
        for ns in ${NS[@]} ; do
            echo "Processing for Thickness = $tk | Overlap = $overlap | NS = $ns"
            FOLDER="${MAPPED_FOLDER}/ADNI_FS_mapped_tk_${tk}_overlap_${overlap}_ns_${ns}/curvelet"

            if [[ -d "$FOLDER" ]]; then
                CMD="python3 ${SCRIPT} -folder ${FOLDER} -tk ${tk} -overlap ${overlap} -ns ${ns}"
                eval ${CMD}
            fi
        done
    done
done
