#!/usr/bin/env python3

from typing import List
import traceback

import logging
import logging.config
import argparse
from datetime import datetime, date, timedelta
import sys
from pathlib import Path
import sqlite3

import anobbsclient
from anobbsclient.walk import create_walker, BoardWalkTarget

from commons import local_tz, client
from commons.updating_model import DB

logging.config.fileConfig('logging.2.5_check_status_of_threads.conf')


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'since', type=str, nargs='?',
        help='\n'.join([
            "检查截止到的日期或日期+时间，格式为 RFC 3339。",
            "省缺则为四个小时前的前一天的上午4时",
        ]),
    )
    parser.add_argument(
        '--board-id', type=int, required=True, dest='board_id',
        help="版块 ID",
    )
    parser.add_argument(
        '--db-path', type=Path, required=True, dest='db_path',
        help="DB 所在的路径",
    )

    parsed = parser.parse_args(args)

    if parsed.since is None:
        parsed.since = ((datetime.now(tz=local_tz) - timedelta(hours=4)).date()
                        - timedelta(days=1)).isoformat()
    parsed.since = parsed.since.strip()
    if 'T' in parsed.since or ' ' in parsed.since:
        parsed.since = datetime.fromisoformat(
            parsed.since).replace(tzinfo=local_tz)
    else:
        parsed.since = datetime.fromisoformat(
            f'{parsed.since} 04:00:00').replace(tzinfo=local_tz)

    return parsed


def main():

    args = parse_args(sys.argv[1:])
    if not args.db_path.exists():
        logging.critical(f"{args.db_path} 不存在，将终止")
        exit(1)

    with sqlite3.connect(args.db_path) as conn:
        db = DB(conn=conn)

        seen_thread_ids = set(db.get_thread_ids_seen_after(args.since))

        walker = create_walker(
            target=BoardWalkTarget(
                start_page_number=1,
                board_id=args.board_id,
                stop_before_datetime=args.since,
            ),
            client=client,
        )

        try:
            for (_, page, _) in walker:
                page: List[anobbsclient.BoardThread] = page
                now = datetime.now(tz=local_tz)

                for thread in page:
                    seen_thread_ids.discard(thread.id)
                    db.record_thread(thread)
                    db.report_is_thread_disappeared(thread.id, now, False)

            for not_found_thread_id in seen_thread_ids:
                # 只若先前没有发现消失，才会对此更新
                if not db.is_thread_disappeared(not_found_thread_id):
                    logging.info(f"发现 {not_found_thread_id} 消失")
                    db.report_is_thread_disappeared(
                        not_found_thread_id, now, True)

        except Exception as e:
            logging.critical(traceback.format_exc())
            raise e

    logging.info("Done")


if __name__ == '__main__':
    main()
