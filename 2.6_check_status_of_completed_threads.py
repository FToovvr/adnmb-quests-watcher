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

import anobbsclient

from commons import get_client
from commons.updating_model import DB, Stats
from commons.stat_model import DB as StatModelDB

logging.config.fileConfig('logging.2.6_check_status_of_completed_threads.conf')

COMPLETION_REGISTRY_THREAD_ID = 22762342


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查登记了完结的串是否更新了蓝字。"
    )

    parser.add_argument(
        '--db-path', type=Path, required=True, dest='db_path',
        help="DB 所在的路径",
    )

    parsed = parser.parse_args(args)

    return parsed


def main():

    args = parse_args(sys.argv[1:])
    if not args.db_path.exists():
        logging.critical(f"{args.db_path} 不存在，将终止")
        exit(1)

    client = get_client()

    with sqlite3.connect(args.db_path) as conn:

        db = DB(conn=conn)

        stats = Stats()

        message = None
        is_successful = True
        try:
            scan_finished_threads(args, db, client, stats)
        except:
            exc_text = traceback.format_exc()
            logging.critical(exc_text)
            message = exc_text
            is_successful = False
        finally:
            db.report_end(is_successful, message, stats)

    if is_successful:
        logging.info("成功结束")
    else:
        exit(1)


def scan_finished_threads(_: argparse.Namespace, db: DB, client: anobbsclient.Client, stats: Stats):
    # 挑出登记完结的串中，尚未蓝字标记完结、今日未曾出现、未被删除的串。
    # 扫描这些串，防止晚标了蓝字导致串被漏掉。
    # 不过没有登记还被晚标的话就没辙了

    # XXX: 不加锁或是进事务了，反正也不会多线程
    db.conn.execute(r'''
        CREATE TEMPORARY TABLE new_completion_registry_entry (
            post_id             INTEGER,
            subject_thread_id   INTEGER
        )
    ''')
    db.conn.execute(r'''
        CREATE INDEX temp.idx__new_completion_registry_entry__subject_thread_id
            ON new_completion_registry_entry(subject_thread_id)
    ''')

    for [id, content] in db.conn.execute(r'''
            SELECT id, content
            FROM post
            WHERE parent_thread_id = ?
                AND id > coalesce((SELECT max(post_id) FROM completion_registry_entry), 0)
        ''', (COMPLETION_REGISTRY_THREAD_ID,)):
        subject_thread_ids = re.findall(r'No\.(\d+)', content)
        for subject_thread_id in list(map(lambda x: int(x), subject_thread_ids)):
            db.conn.execute(r'''
                INSERT INTO new_completion_registry_entry (post_id, subject_thread_id)
                VALUES (?, ?)
            ''', (id, subject_thread_id))

    db.conn.create_function("extract_blue_text", 1,
                            StatModelDB.extract_blue_text)

    db.conn.execute(r'''
        INSERT INTO completion_registry_entry (post_id, subject_thread_id, has_blue_text_been_added)
        SELECT
            post_id,
            subject_thread_id,
            (SELECT extract_blue_text(content) FROM thread WHERE id = subject_thread_id) IS NOT NULL
        FROM new_completion_registry_entry
    ''')

    db.conn.execute(r'DROP TABLE new_completion_registry_entry')
    db.conn.commit()

    for [id] in db.conn.execute(r'''
        SELECT DISTINCT subject_thread_id
        FROM completion_registry_entry
        LEFT JOIN thread ON subject_thread_id = thread.id
        WHERE has_blue_text_been_added = FALSE
    '''):
        [page, usage] = client.get_thread_page(id=id, page=1,
                                               for_analysis=True)
        stats.thread_request_count += 1
        stats.total_bandwidth_usage.add(usage)
        db.record_thread(page)
        blue_text = StatModelDB.extract_blue_text(page.content)
        logging.info(f"串 {id} 当前蓝字：{blue_text}")
        if blue_text is not None:
            db.conn.execute(r'''
                UPDATE completion_registry_entry
                SET has_blue_text_been_added = TRUE
                WHERE subject_thread_id = ?
            ''', (id,))
    db.conn.commit()


if __name__ == '__main__':
    main()
