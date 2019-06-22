#!/usr/bin/env bash
# Parse args
MSG=${1-"Notification from ${HOSTNAME}"}

# Get current dir and notification script
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NOTIFIER="${CURRENT_DIR}/../lib/telegram_notifier.py"

CMD="${NOTIFIER} -msg ${MSG}"
eval ${CMD}
