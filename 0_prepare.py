#!/usr/bin/env python3

from datetime import datetime, timedelta
import os
import subprocess

LOG_FILE_PATH_FORMAT = 'logs/%Y-%m-%d'

today = datetime.today()
yesterday = today - timedelta(days=1)

os.makedirs(today.strftime(LOG_FILE_PATH_FORMAT), exist_ok=True)

yesterday_log_folder = yesterday.strftime(LOG_FILE_PATH_FORMAT)
if os.path.isdir(yesterday_log_folder):
    subprocess.run(
        ['tar', 'czf', f'{yesterday_log_folder}.tgz', yesterday_log_folder])
