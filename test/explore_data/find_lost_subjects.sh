#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MISSED_LIST=($(cat "${CURRENT_DIR}/missed_subjects.txt"))
#TARGET_DIR="/run/media/ssilvari/Smith_2T_WD/Databases/"
TARGET_DIR="/disk/"

SIN_ZIP=($(find ${TARGET_DIR} -type f -name "*.zip" -exec unzip -Z1 {} \; | grep "_S_" | grep ".nii"))
#SIN_ZIP=($(find ${TARGET_DIR} -type d -name "*_S_*"))

for f in ${SIN_ZIP[@]}
do
    for  subject in ${MISSED_LIST[@]}
    do
        # echo -e "\nFinding ${subject} in ${f}"
        if echo "${f}" | grep -q "${subject}";
        then
            echo "Subject ${subject} found in " ${f}
            echo ${subject} >> "/home/ssilvari/Documents/temp/ADNI_test/ADNI_FS_sphere_mapped/groupfile.csv"
        fi
    done
done
