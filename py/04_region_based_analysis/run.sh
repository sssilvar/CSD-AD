#!/usr/bin/env bash

# Get current dir
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run feature extraction
eval "python "${CURRENT_DIR}"/region_feature_extraction.py"

# Run Classification
cmd="python "${CURRENT_DIR}"/relevant_regions.py > "${CURRENT_DIR}"/../../output/gmm_region_based.log"
echo -e "\n\n[  INFO  ] Running classification stage"
eval ${cmd}