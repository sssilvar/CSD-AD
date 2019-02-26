#!/usr/bin/env bash

# Get current directory
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set conversion times of interest
TIMES=(24 36 60)

# Set other params
CLF="rf"
FOLDS=10
IMG_TYPES=("intensity" "gradient" "sobel")

for imtype in ${IMG_TYPES[@]}
do
    for t in ${TIMES[@]}
    do
        echo "Processing ${imtype} images for subjects in ${t} months of conversion/stability..."
        SCRIPT="${CURRENT_DIR}/classify_complete.py -time ${t} -folds ${FOLDS} -clf ${CLF} -imtype ${imtype}"
        eval "python3 ${SCRIPT}"

        echo -e "\n\n"
    done
done
