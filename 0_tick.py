#!/usr/bin/env python3

from datetime import datetime, timedelta, date
import os
import subprocess

import psycopg2

from commons.consts import local_tz, get_target_date
from commons.config import load_config
from models.activity import Activity
from models.publication_record import PublicationRecord

LOG_FILE_PATH_FORMAT = 'logs/%Y-%m-%d'


def round_to_minutes(datetime: datetime) -> datetime:
    return datetime.replace(second=0, microsecond=0)


def main():
    now = round_to_minutes(datetime.now(tz=local_tz))
    today = now.date()

    prepare_log_folder(today=today)

    config = load_config('./config.yaml')

    with psycopg2.connect(config.database.connection_string) as conn:
        conn: psycopg2._psycopg.connection = conn

        # 每五分钟采集一次
        if now.minute % 5 == 0:
            last_run_at_collect = round_to_minutes(
                Activity.get_last_activity_run_at(conn, 'collect'))
            if now >= last_run_at_collect + timedelta(minutes=5):
                result = subprocess.run('./1_collect.py')
                assert(result.returncode == 0)

        # 每单数小时 55 分检查一次完结情况
        if now.hour % 2 == 1 and now.minute >= 55:
            last_run_at_check_completed = round_to_minutes(
                Activity.get_last_activity_run_at(conn, 'check_completed'))
            if now >= last_run_at_check_completed + timedelta(hours=1):
                result = subprocess.run([
                    './2.6_check_completed.py',
                ])
                assert(result.returncode == 0)

        target_date = get_target_date(now)
        # 报告中午 12 点后发
        if now.hour >= 12 and \
                not PublicationRecord.is_report_published(
                    conn=conn, subject_date=target_date, report_type='trend'):
            # 先检查一下报告的那天有没有那些串消失了
            last_run_at_check_disappeared = round_to_minutes(
                Activity.get_last_activity_run_at(conn, 'check_disappeared'))
            # 为了防止发送失败导致此步骤频繁执行，每次执行间隔至少 1 小时
            if now >= last_run_at_check_disappeared + timedelta(hours=1):
                result = subprocess.run([
                    './2.5_check_disappeared.py',
                    target_date.isoformat(),
                ])
                assert(result.returncode == 0)

            # 发布报告
            result = subprocess.run([
                './3_generate_text_report.py', target_date.isoformat(),
                '--check-sage',
                '--publish', '--notify-daily-qst',
            ])
            assert(result.returncode == 0)


def prepare_log_folder(today: date):
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


if __name__ == '__main__':
    main()
