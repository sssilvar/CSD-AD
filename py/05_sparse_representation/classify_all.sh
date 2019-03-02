#!/usr/bin/env bash

# Get current directory
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set conversion times of interest
TIMES=(24 36 60)

# Features file
FEATS="/home/jullygh/sssilvar/Documents/Dataset/ADNI_FS_sphere_mapped/curvelets_non_split"

# Set other params
CLF="svm"
FOLDS=5
IMG_TYPES=("gradient" "sobel")

for imtype in ${IMG_TYPES[@]}
do
    for t in ${TIMES[@]}
    do
        echo "Processing ${imtype} images for subjects in ${t} months of conversion/stability..."
        SCRIPT="${CURRENT_DIR}/classify_complete.py -time ${t} -folds ${FOLDS} -clf ${CLF} -imtype ${imtype} -tune 1 -features ${FEATS}/${imtype}_curvelet_features_non_split.csv"
        eval "python3 ${SCRIPT}"

        echo -e "\n\n"
    done
done
