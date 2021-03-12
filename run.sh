#!/usr/bin/env bash

set -e

# cd 到脚本所在目录
cd "$(dirname "${BASH_SOURCE[0]}")"

# exec flock -n .lock

#source ./venv/bin/activate
source ./export-env.sh
source ./export-secrets.sh

./0_run.py
