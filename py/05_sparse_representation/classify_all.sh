#!/usr/bin/env bash

# Get current directory
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set conversion times of interest
TIMES=(24 36 60)

# Features file
FEATS=${1-"/home/jullygh/sssilvar/Documents/Dataset/ADNI_FS_sphere_mapped_6_sph_overlap/curvelets_non_split"}
NBS=${2-"4"}
NBA=${3-"32"}

# Set other params
CLASSIFIERS=("svm" "rf")
FOLDS=7
IMG_TYPES=("gradient" "sobel")

for clf in ${CLASSIFIERS[@]}
do
    for imtype in ${IMG_TYPES[@]}
    do
        for t in ${TIMES[@]}
        do
            echo "Processing ${imtype} images for subjects in ${t} months of conversion/stability..."
            FEATS_FILE="${FEATS}/${imtype}_curvelet_features_non_split_${NBS}_scales_${NBA}_angles.csv"
            SCRIPT="${CURRENT_DIR}/splitting_images/classify_complete.py -time ${t} -folds ${FOLDS} -clf ${clf} -imtype ${imtype} -tune 0 -features ${FEATS_FILE}"
            eval "python3 ${SCRIPT}"

            echo -e "\n\n"
        done
    done
done
