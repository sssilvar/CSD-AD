#!/usr/bin/env bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SCRIPT="${CURRENT_DIR}/../py/04_mapping_sphere/sphere_mapping_indexes_sql.py"
NOTIFIER="${CURRENT_DIR}/../lib/email_notifier.py -subject 'Index calculation finished'"

echo "Executing ${SQL_INDEX_GEN} ..."

# Thicknesses and overlaps
#TKS=(15 20 30)
#OVERLAPS=(4 5 7)

# Test only
TKS=(5641)
OVERLAPS=(6151)

for tk in ${TKS[@]}
do
    for overlap in ${OVERLAPS[@]}
    do
        echo "Processing for tk = ${tk} and overlap = ${overlap}"
        NOTIFIER_EMAIL="${NOTIFIER} -msg 'Index calculation for tk = ${tk} and overlap = ${overlap} is finished.'"
        CMD="${SCRIPT} ${tk} ${overlap} && ${NOTIFIER_EMAIL}"

        CMD="tmux new-session -d -s \"sql_ix_${tk}_ov_${overlap}\" \"${CMD}\""
        echo ${CMD}
        eval "${CMD}"
        echo -e "Done!\n\n"
    done
done

# List running processes
eval "tmux ls"
