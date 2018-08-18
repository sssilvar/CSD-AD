#!/bin/bash
DATA_FOLDER=$1
SCALES=$2
ANGLES=$3

# IMAGE INFO
CONTAINER_NAME="neuro_curvelets"
USER="sssilvar"

IMG_NAME=$USER"/"${CONTAINER_NAME}
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS_DIR=${CURRENT_DIR}"/../"

echo -e "PIPELINE INFO:\n\t- N. Scales: "${SCALES}"\n\t- N. Angles: "${ANGLES}

# RUN CONTAINER
echo -e "\n\n[  OK  ] Running container: "${CONTAINER_NAME}
CMD="docker run --name "${CONTAINER_NAME}" --rm -ti -v "${DATA_FOLDER}":/root/data/ -v "${SCRIPTS_DIR}":/py -e 'SCALES="${SCALES}"' -e 'ANGLES="${ANGLES}"' "${IMG_NAME}
echo ${CMD}
eval ${CMD}