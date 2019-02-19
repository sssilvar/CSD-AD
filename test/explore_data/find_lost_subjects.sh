#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MISSED_LIST=($(cat "${CURRENT_DIR}/missed_subjects.txt"))
#TARGET_DIR="/run/media/ssilvari/Smith_2T_WD/Databases/"
TARGET_DIR="/disk/data/ADNI/ADNI_2_3_GO_SPGR/"
GFILE="/home/ssilvari/Documents/temp/ADNI_test/ADNI_FS_sphere_mapped/groupfile_missed.csv"

SIN_ZIP=($(find ${TARGET_DIR} -type f -name "*.nii" | grep "_S_"))
#SIN_ZIP=($(find ${TARGET_DIR} -type d -name "*_S_*"))

rm ${GFILE}
echo "subj" > ${GFILE}

for f in ${SIN_ZIP[@]}
do
    for  subject in ${MISSED_LIST[@]}
    do
        # echo -e "\nFinding ${subject} in ${f}"
        if echo "${f}" | grep -q "${subject}";
        then
            for i in {1..4}
            do
                f=$(dirname ${f})
            done
            echo "Subject ${subject} found in " ${f}
            scp -r ${f} umng:/home/jullygh/sssilvar/Documents/Dataset/ADNI
            echo ${subject} >> ${GFILE}
        fi
    done
done

scp ${GFILE} umng:/home/jullygh/sssilvar/Documents/Dataset/ADNI