#!/usr/bin/env bash
current_dir=$(pwd -P)
echo "[  INFO  ] Current directory: $current_dir"

file="$current_dir/../py/conversion_detect_vol_tk.py"
echo "[  OK  ] Executing file:  $file"

set -x
/usr/bin/python2.7 $file
