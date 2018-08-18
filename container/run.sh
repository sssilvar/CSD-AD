#!/bin/bash
DATA_FOLDER=$1
SCALES=$2
ANGLES=$3

echo -e "PIPELINE INFO:\n\t- N. Scales: "${SCALES}"\n\t- N. Angles: "${ANGLES}

echo -e "\n\n[  OK  ] Running container: "${CONTAINER_NAME}
CMD="docker run --name "${CONTAINER_NAME}" --rm -ti -v "${DATA_FOLDER}":/root/data/ -v "${SCRIPTS_DIR}":/py -e 'SCALES="${SCALES}"' -e 'ANGLES="${ANGLES}"' "${IMG_NAME}
echo ${CMD}
eval ${CMD}