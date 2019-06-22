#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INVENTORY_FILE="${CURRENT_DIR}/inventory"
PLAYBOOK="${CURRENT_DIR}/classify_all.yml"

# Check if inventory file exists
if [[ ! -f ${INVENTORY_FILE} ]]; then
    echo "[ ERROR ] Inventory file not found."
    echo "Create inventory file at: ${INVENTORY_FILE}"
    exit 1
fi

## Perform a ping
#CMD="ansible umng -i ${INVENTORY_FILE} -m ping -vvv"
#eval ${CMD}
#
## Check python version
#CMD="ansible umng -i ${INVENTORY_FILE} -a \"python --version\""
#eval ${CMD}
