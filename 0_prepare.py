#!/usr/bin/env python3

from datetime import datetime, timedelta
import os
import subprocess

from dateutil import tz

local_tz = tz.gettz('Asia/Shanghai')

LOG_FILE_PATH_FORMAT = 'logs/%Y-%m-%d'


def main():
    today = datetime.now(tz=local_tz)
    yesterday = today - timedelta(days=1)

    # 如果不存在，创建今日的日志文件夹
    os.makedirs(today.strftime(LOG_FILE_PATH_FORMAT), exist_ok=True)

    # 如果存在，归档昨日的日志
    yesterday_log_folder = yesterday.strftime(LOG_FILE_PATH_FORMAT)
    if os.path.isdir(yesterday_log_folder):
        result = subprocess.run(
            ['tar', 'czf', f'{yesterday_log_folder}.tgz', yesterday_log_folder])
        if result.returncode == 0:
            subprocess.run(['command', 'rm', '-rf', yesterday_log_folder])


if __name__ == '__main__':
    main()
