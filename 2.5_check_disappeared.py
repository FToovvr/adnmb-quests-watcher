#!/usr/bin/env python3

from typing import List
from dataclasses import dataclass
import traceback

import logging
import logging.config
import argparse
from datetime import datetime, date, timedelta
import sys
from pathlib import Path

import psycopg2

import anobbsclient
from anobbsclient.walk import create_walker, BoardWalkTarget

from commons.consts import local_tz, get_target_date
from commons.config import load_config, ClientConfig
from models.activity import Activity, Stats
from models.collecting import DB

logging.config.fileConfig('logging.2.5_check_status_of_threads.conf')


@dataclass(frozen=True)
class Arguments:
    config_file_path: str

    since: datetime
    board_id: int
    completion_registry_thread_id: int  # 其实不需要

    connection_string: str

    client_config: ClientConfig


def parse_args(args: List[str]) -> Arguments:
    parser = argparse.ArgumentParser(
        description="检查截止到指定位置活动过的主题串是否仍然存在。"
    )

    parser.add_argument(
        '-c', '--config', type=str, default='./config.yaml',
        dest='config_file_path',
        help='配置文件路径',
    )

    parser.add_argument(
        'since', type=str, nargs='?',
        help='\n'.join([
            "截止到的日期或日期+时间，格式为 RFC 3339。",
            "省缺则为四个小时前的前一天的上午4时",
        ]),
    )

    parsed = parser.parse_args(args)
    config = load_config(parsed.config_file_path)

    if parsed.since is None:
        parsed.since = get_target_date().isoformat()
    parsed.since = parsed.since.strip()
    if 'T' in parsed.since or ' ' in parsed.since:
        parsed.since = datetime.fromisoformat(
            parsed.since).replace(tzinfo=local_tz)
    else:
        parsed.since = datetime.fromisoformat(
            f'{parsed.since} 04:00:00').replace(tzinfo=local_tz)

    return Arguments(
        config_file_path=parsed.config_file_path,

        since=parsed.since,
        board_id=config.board_id,
        completion_registry_thread_id=config.completion_registry_thread_id,
        connection_string=config.database.connection_string,
        client_config=config.client,
    )


def main():

    args = parse_args(sys.argv[1:])

    client = args.client_config.create_client()

    with psycopg2.connect(args.connection_string) as conn_activity, \
            psycopg2.connect(args.connection_string) as conn_db:
        activity = Activity(conn=conn_activity,
                            activity_type='check_disappeared',
                            run_at=datetime.now(tz=local_tz))
        db = DB(conn=conn_db,
                completion_registry_thread_id=args.completion_registry_thread_id)

        stats = Stats()

        message = None
        is_successful = True
        try:
            rescan_board(args, db, client, stats)
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


def rescan_board(args: argparse.Namespace, db: DB, client: anobbsclient.Client, stats: Stats):
    # 用于检测当天消失的串，
    # 但如果当天消失的串最后上浮的时间在当天之前，就无法检测到了

    thread_ids_seen_today = set(db.get_thread_ids_seen_since(args.since))

    walker = create_walker(
        target=BoardWalkTarget(
            start_page_number=1,
            board_id=args.board_id,
            stop_before_datetime=args.since,
        ),
        client=client,
    )

    for (_, page, usage) in walker:
        page: List[anobbsclient.BoardThread] = page
        now = datetime.now(tz=local_tz)
        stats.board_request_count += 1
        stats.total_bandwidth_usage.add(usage)

        for thread in page:
            thread_ids_seen_today.discard(thread.id)
            db.record_thread(thread, board_id=args.board_id, updated_at=now)
            db.report_is_thread_disappeared(thread.id, now, False)

    for not_found_thread_id in thread_ids_seen_today:
        # 只若先前没有发现消失，才会对此更新
        if not db.is_thread_disappeared(not_found_thread_id):
            logging.info(f"发现 {not_found_thread_id} 消失")
            db.report_is_thread_disappeared(
                not_found_thread_id, now, True)


if __name__ == '__main__':
    main()
