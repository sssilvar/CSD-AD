#!/bin/bash

#====FOLDER====
DATA_FOLDER=$1
SCALES=$2
ANGLES=$3

echo -e "PIPELINE INFO:\n\t- N. Scales: "${SCALES}"\n\t- N. Angles: "${ANGLES}

# Set parameters up
CONTAINER_NAME="neuro_curvelets"
USER="sssilvar"

IMG_NAME=$USER"/"${CONTAINER_NAME}
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPTS_DIR=${CURRENT_DIR}"/../"

echo -e "\n\n[  OK  ] Stoping container"
STOP_CONT="docker stop "${CONTAINER_NAME}
eval ${STOP_CONT}

echo -e "\n\n[  OK  ] Deleting container"
DEL_CONT="docker rm "${CONTAINER_NAME}
eval ${DEL_CONT}

echo -e "\n\n[  OK  ] Deleting image"
DEL_IMG="docker rmi "${IMG_NAME}
eval ${DEL_IMG}

echo -e "\n\n[  OK  ] Creating the new image: "${IMG_NAME}
CRE_IMG="docker build -t "${IMG_NAME}" --build-arg proxy="${PROXY}" "${CURRENT_DIR}
eval ${CRE_IMG}

echo -e "\n\n[  OK  ] Running container: "${CONTAINER_NAME}
CMD="docker run --name "${CONTAINER_NAME}" --rm -ti -v "${DATA_FOLDER}":/root/data/ -v "${SCRIPTS_DIR}":/py -e 'SCALES="${SCALES}"' -e 'ANGLES="${ANGLES}"' "${IMG_NAME}
echo ${CMD}
eval ${CMD}