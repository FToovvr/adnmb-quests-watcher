#!/usr/bin/env bash

set -e

#source ./venv/bin/activate
source ./export-env.sh
source ./export-secrets.sh

# TODO: 每5分钟抓取，每1小时生成24小时报告，每1天生成归档报告

./1_update_database.py