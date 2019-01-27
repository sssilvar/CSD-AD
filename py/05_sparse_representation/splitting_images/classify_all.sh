#!/bin/bash

# Get current directory
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR=${1-"/home/ssilvari/Documents/temp/spherical_mapping/sphere_mapped_4_spheres"}
ROC_FOLDER="${MAIN_DIR}/ROC"
echo -e "[  INFO  ] Data folder in: ${CURRENT_DIR}"

# Define scales and angles
IMG_TYPES=("intensity" "gradient")
SCALES=(4)
ANGLES=(16)

# Create ROC folder if does not exist
if [ ! -d ${ROC_FOLDER} ]; then
    echo "[  INFO  ] Creating Results Folder: ${ROC_FOLDER}"
    mkdir $ROC_FOLDER
fi

# Run classification
for img_type in ${IMG_TYPES[@]}
do
    for scale in ${SCALES[@]}
    do
        for angle in ${ANGLES[@]}
        do
            SCRIPT="${CURRENT_DIR}/classification_per_sphere.py"
            CSV_FILE="${MAIN_DIR}/${img_type}_curvelet_features_${scale}_scales_${angle}_angles.csv"
            CMD="python3 ${SCRIPT} ${CSV_FILE}"

            # Check if CSV exists and run
            if [ -f ${CSV_FILE} ]; then
                echo -e "[  INFO  ]Running classification..."
                echo -e "\t- Image type: ${img_type}"
                echo -e "\t- Number of scales: ${scale}"
                echo -e "\t- Number of orientations: ${angle}"
                eval "${CMD} > ${ROC_FOLDER}/classification_${scale}_scales_${angle}_angles_${img_type}.log"
            else
                echo -e "[  ERROR  ] File ${CSV_FILE} not found!\n"
            fi
        done
    done
done