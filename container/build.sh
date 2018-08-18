#!/bin/bash

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