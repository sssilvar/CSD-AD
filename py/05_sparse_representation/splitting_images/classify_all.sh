#!/bin/bash

# Get current directory
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MAIN_DIR=${1-"/home/ssilvari/Documents/temp/spherical_mapping/sphere_mapped_4_spheres"}
echo -e "[  INFO  ] Data folder in: ${CURRENT_DIR}"

# Define scales and angles
SCALES=(4)
ANGLES=(16)

# Run classification
for scale in ${SCALES[@]}
do
    for angle in ${ANGLES[@]}
    do
        SCRIPT="${CURRENT_DIR}/classification_per_sphere.py"
        CSV_FILE="${MAIN_DIR}/intensity_curvelet_features_${scale}_scales_${angle}_angles.csv"
        CMD="python3 ${SCRIPT} ${CSV_FILE}"

        # Check if CSV exists and run
        if [ -f ${CSV_FILE} ]; then
            echo -e "$CMD\n"
            eval "${CMD} > ${MAIN_DIR}/ROC/classification_${scale}_scales_${angle}_angles.log"
        else
            echo -e "[  ERROR  ] File ${CSV_FILE} not found!\n"
        fi
    done
done