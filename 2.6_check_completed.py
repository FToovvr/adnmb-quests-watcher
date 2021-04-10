#!/usr/bin/env python3

from typing import List
import traceback

import logging
import logging.config
import argparse
from datetime import datetime, date
import sys
from pathlib import Path
import re

import psycopg2

import anobbsclient

from commons.consts import local_tz
from commons.config import load_config
from models.activity import Activity, Stats
from models.collecting import DB

logging.config.fileConfig('logging.2.6_check_status_of_completed_threads.conf')

# FIXME: 遇到被删的串不会记录被删，导致会一直检查下去


def main():

    config = load_config('./config.yaml')

    client = config.client.create_client()

    with psycopg2.connect(config.database.connection_string) as conn_activity, \
            psycopg2.connect(config.database.connection_string) as conn_db:
        activity = Activity(conn=conn_activity,
                            activity_type='check_completed',
                            run_at=datetime.now(tz=local_tz))
        db = DB(conn=conn_db,
                completion_registry_thread_id=config.completion_registry_thread_id)

        stats = Stats()

        message = None
        is_successful = True
        try:
            scan_finished_threads(db, client, stats)
        except:
            exc_text = traceback.format_exc()
            logging.critical(exc_text)
            message = exc_text
            is_successful = False
        finally:
            activity.report_end(is_successful, message, stats)

    if is_successful:
        logging.info("成功结束")
    else:
        exit(1)


def scan_finished_threads(db: DB, client: anobbsclient.Client, stats: Stats):
    # 挑出登记完结的串中，尚未蓝字标记完结、今日未曾出现、未被删除的串。
    # 扫描这些串，防止晚标了蓝字导致串被漏掉。
    # 不过没有登记还被晚标的话就没辙了

    for id in db.get_thread_ids_in_completion_registry_thread_without_blue_texts():
        [page, usage] = client.get_thread_page(id=id, page=1,
                                               for_analysis=True)
        stats.total_bandwidth_usage.add(usage, int(page._raw['fid']),
                                        run_at=datetime.now(tz=local_tz))
        db.record_thread(thread=page)


if __name__ == '__main__':
    main()
