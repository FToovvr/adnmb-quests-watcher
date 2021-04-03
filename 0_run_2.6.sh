#!/usr/bin/env bash

set -e

# cd 到脚本所在目录
cd "$(dirname "${BASH_SOURCE[0]}")"

# exec flock -n .lock

./2.6_check_status_of_completed_threads.py --db-path db.sqlite3