from typing import Any, Optional,  List, Dict
from dataclasses import dataclass, field

import logging
from datetime import datetime, timedelta
import json
import traceback

import psycopg2

import anobbsclient

# pylint: disable=relative-beyond-top-level
from ..commons.consts import local_tz


@dataclass(frozen=True)
class DB:

    conn: psycopg2._psycopg.connection

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger('DB'))

    def try_find_thread_latest_seen_reply_id(self, thread_id: int) -> Optional[int]:
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''SELECT find_thread_latest_seen_response_id(%s)''',
                        (thread_id,))
            row = cur.fetchall()
        if row is None:
            return None

        return row[0]

    def record_thread(self, thread: anobbsclient.ThreadPage, board_id: int, record_total_reply_count: bool = True):
        self.logger.info(f'正在记录/更新串信息。串号 = {thread.id}')
        updated_at = _updated_at,

        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'CALL record_thread(' + ', '.join(['%s' * 13]) + r')', (
                thread.id, board_id, thread.created_at, thread.user_id, thread.content,
                thread.attachment_base, thread.attachment_extension,
                thread.name, thread.email, thread.title, DB.__extract_misc_fields(thread)),
                thread.total_reply_count, updated_at,
            )

        self.logger.info(f'已记录/更新串信息。串号 = {thread.id}')

    def record_thread_replies(self, thread: anobbsclient.BoardThread,
                              replies: List[anobbsclient.Post], total_reply_count: int):
        """
        记录串中新抓取到的回应。

        之前的实现是开一个事务一次性把收集到的回应一起灌进去，
        不过其实保持按时间顺序加入挨个就好，不用事务

        Parameters
        ----------
        thread : anobbsclient.ThreadPage
            来自版块页面的串首。
        replies : List[anobbsclient.Post]
            本次要记录的全部回应。
        total_reply_count : int
            串回应总数。

            单独列出来是因为抓取途中可能会有变化，采用抓取到最后时的数字。
        """
        self.logger.info(f'正在记录串中新抓取到的回应。串号 = {thread.id}，'
                         + f'新记录回应数 = {len(replies)}，更新的回应总数 = {total_reply_count}')

        replies = sorted(replies, key=lambda r: r.id)

        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur

            cur.execute(r'CALL update_thread_extra_current_reply_count(%s, %s)',
                        (thread.id, total_reply_count))

            for post in replies:
                cur.execute(r'CALL record_response(' + ', '.join(['%s' * 12]) + r')', (
                    post.id, thread.id, post.created_at, post.user_id, post.content,
                    post.attachment_base, post.attachment_extension,
                    post.name, post.email, post.title, DB.__extract_misc_fields(post)),
                    updated_at,
                )

        self.logger.info(f'已记录串中新抓取到的回应。串号 = {thread.id}')

    def get_thread_total_reply_count(self, thread_id: int) -> int:
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT get_thread_total_response_count(%s)',
                        (thread_id,))
            return cur.fetchone()[0]

    def is_thread_recorded(self, thread_id: int) -> bool:
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT is_thread_in_database(%s)',
                        (thread_id,))
            return cur.fetchone()[0]

    @ staticmethod
    def __extract_misc_fields(post: anobbsclient.Post) -> Optional[Dict[str, Any]]:
        post_raw: Dict[str, Any] = post.raw_copy()
        for key in ['id', 'img', 'ext', 'now', 'userid', 'name', 'email', 'title', 'content',
                    'replys', 'replyCount', 'remainReplys']:
            post_raw.pop(key, None)
        if not post.marked_sage:
            post_raw.pop('sage', None)
        if not post.marked_admin:
            post_raw.pop('admin', None)
        if post_raw.get('status', None) == 'n':
            post_raw.pop('status')
        post_raw.pop('fid', None)
        if len(post_raw) == 0:
            return None
        return json.dumps(post_raw)

    def get_thread_ids_seen_since(self, datetime: datetime) -> List[int]:
        """
        获取曾在指定时间后看到过的各串的串号
        """
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT get_thread_ids_seen_since(%s)',
                        (datetime,))
            return cur.fetchone()[0]

    def is_thread_disappeared(self, thread_id: int) -> bool:
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT is_thread_disappeared(%s)',
                        (thread_id,))
            return cur.fetchone()[0]

    def report_is_thread_disappeared(self, thread_id: int, checked_at: datetime, value: bool):
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'CALL update_thread_extra_is_disappeared(%s, %s, %s)',
                        (thread_id, checked_at, value))
