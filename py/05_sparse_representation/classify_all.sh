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
FOLDS=10
IMG_TYPES=("gradient" "sobel")

for clf in ${CLASSIFIERS[@]}
do
    for imtype in ${IMG_TYPES[@]}
    do
        for t in ${TIMES[@]}
        do
            echo "Processing ${imtype} images for subjects in ${t} months of conversion/stability..."
            FEATS_FILE="${FEATS}/${imtype}_curvelet_features_non_split_${NBS}_scales_${NBA}_angles_norm.csv"
            SCRIPT="${CURRENT_DIR}/splitting_images/classify_complete.py -time ${t} -folds ${FOLDS} -clf ${clf} -imtype ${imtype} -tune 0 -features ${FEATS_FILE}"
            if [[ -f $FEATS_FILE ]]; then
                eval "python3 ${SCRIPT}"
            else
                echo "File ${FEATS_FILE} not found"
            fi

            echo -e "\n\n"
        done
    done
done
