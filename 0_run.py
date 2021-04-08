#!/usr/bin/env python3

from datetime import datetime, timedelta
import os
import subprocess

from commons import local_tz, Trace, get_target_date

LOG_FILE_PATH_FORMAT = 'logs/%Y-%m-%d'

# TODO: 每5分钟抓取，每1小时生成24小时报告，每1天生成归档报告


def main():
    now = datetime.now(tz=local_tz)
    target_date = get_target_date(now)
    today = now.date()
    yesterday = today - timedelta(days=1)

    # 如果不存在，创建今日的日志文件夹
    os.makedirs(today.strftime(LOG_FILE_PATH_FORMAT), exist_ok=True)

    # 如果存在，归档昨日的日志
    yesterday_log_folder = yesterday.strftime(LOG_FILE_PATH_FORMAT)
    if os.path.isdir(yesterday_log_folder):
        result = subprocess.run(
            ['tar', 'czf', f'{yesterday_log_folder}.tgz', yesterday_log_folder])
        if result.returncode == 0:
            subprocess.run(['/bin/rm', '-rf', yesterday_log_folder])

    result = subprocess.run('./1_collect.py')
    assert(result.returncode == 0)

    with sqlite3.connect('db.sqlite3') as conn:
        has_trace = Trace.has_trace(conn=conn, date=target_date)
        is_report_published = Trace.is_report_published(conn=conn, date=target_date,
                                                        type_='trend')

    if not has_trace:
        result = subprocess.run([
            './2.5_check_status_of_threads.py',
            target_date.isoformat(),
        ])
        assert(result.returncode == 0)

    if not is_report_published:
        result = subprocess.run([
            './3_generate_text_report.py', target_date.isoformat(),
            '--check-sage',
            '--publish', '--notify-daily-qst',
        ])
        assert(result.returncode == 0)


if __name__ == '__main__':
    main()
