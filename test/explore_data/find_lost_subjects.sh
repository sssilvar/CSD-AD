#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MISSED_LIST=($(cat "${CURRENT_DIR}/missed_subjects.txt"))
TARGET_DIR="/run/media/ssilvari/Smith_2T_WD/Databases/"

#SIN_ZIP=($(find ${TARGET_DIR} -type f -name "*.zip" -exec unzip -Z1 {} \; | grep "_S_"))
SIN_ZIP=($(find ${TARGET_DIR} -type d -name "*_S_*"))

for subject in ${MISSED_LIST[@]:1:5}
do
    for f in ${SIN_ZIP[@]}
    do
        if [[ ${subject} ==  *"${f}"* ]];
        then
            echo "Subject ${subject} found in " ${f}
        fi
    done
done
