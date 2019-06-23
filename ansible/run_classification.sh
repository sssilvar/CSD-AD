#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLAYBOOK="${CURRENT_DIR}/classify_all.yml"
INVENTORY="${CURRENT_DIR}/inventory"

CMD="ansible-playbook ${PLAYBOOK} -i ${INVENTORY}"
echo "Running classification pipeline..."
echo ${CMD}
eval ${CMD}
