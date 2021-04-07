from typing import Any, Optional,  List, Dict
from dataclasses import dataclass, field

import sqlite3
import logging
from datetime import datetime, timedelta
import json
import traceback

import anobbsclient

# pylint: disable=relative-beyond-top-level
from ..commons.consts import local_tz


@dataclass(frozen=True)
class DB:

    conn: sqlite3.Connection

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger('DB'))

    def try_find_thread_latest_seen_reply_id(self, thread_id: int) -> Optional[int]:
        row = self.conn.execute(r'''
            SELECT id FROM post WHERE parent_thread_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (thread_id,)).fetchone()
        if row is None:
            return None
        return row[0]

    def record_thread(self, thread: anobbsclient.ThreadPage, record_total_reply_count: bool = True):
        self.logger.info(f'正在记录/更新串信息。串号 = {thread.id}')

        # TODO: 是否考虑 sage？
        old = self.conn.execute(r'''
            SELECT content, name, email, title
            FROM thread WHERE id = ?
        ''', (thread.id,)).fetchone()

        if old is not None \
            and (old[0] != thread.content or old[1] != thread.name
                 or old[2] != thread.email or old[3] != thread.title):
            now = datetime.now(tz=local_tz).timestamp()
            self.logger.info(f'串内容发生变化，将归档当前版本。串号 = {thread.id}，现时时间戳 = {now}')
            self.conn.execute(r'''
                INSERT INTO thread_old_revision (
                    id, not_anymore_at_least_after,
                    content, name, email, title
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                thread.id, now,  # 偷懒
                old[0], old[1], old[2], old[3],
            ))

        # 这样暴力地 replace 就成了，反正每次抓取都会抓取到全部的 fields。
        # ~~不在这里记录 current_reply_count,~~
        # 因为为了准确，需要在抓取完其中的所有新回应时再记录当时的总回应数
        self.conn.execute(r'''
            INSERT OR REPLACE INTO thread (
                id, created_at, user_id, content,
                attachment_base, attachment_extension,
                name, email, title, misc_fields
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?
            )
        ''', (
            thread.id, thread.created_at.timestamp(), thread.user_id, thread.content,
            thread.attachment_base, thread.attachment_extension,
            thread.name, thread.email, thread.title, DB.__extract_misc_fields(thread)),
        )
        if record_total_reply_count:
            # TODO: 应该记在 thread_extra
            self.conn.execute(r'''
                UPDATE thread SET current_reply_count = ?
                WHERE id = ?
            ''', (thread.total_reply_count, thread.id))

        self.logger.info(f'已记录/更新串信息。串号 = {thread.id}')

    def record_thread_replies(self, thread: anobbsclient.BoardThread,
                              replies: List[anobbsclient.Post], total_reply_count: int):
        """
        记录串中新抓取到的回应。

        偷懒，由于实现中继机制比较麻烦，直接一次性提交，中途中断下次就直接从头开始。

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

        cur = self.conn.cursor()
        cur.execute(r'BEGIN')

        self.conn.execute(r'''
            UPDATE thread SET current_reply_count = ?
            WHERE id = ?
        ''', (total_reply_count, thread.id))

        for post in replies:
            # 重复 insert post 是出现了 bug 的征兆
            self.conn.execute(r'''
                INSERT INTO post (
                    id, parent_thread_id, created_at, user_id, content,
                    attachment_base, attachment_extension,
                    name, email, title, misc_fields
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
            ''', (
                post.id, thread.id, post.created_at.timestamp(), post.user_id, post.content,
                post.attachment_base, post.attachment_extension,
                post.name, post.email, post.title, DB.__extract_misc_fields(thread)),
            )

        cur.execute(r'COMMIT')
        cur.close()

        self.logger.info(f'已记录串中新抓取到的回应。串号 = {thread.id}')

    def get_thread_total_reply_count(self, thread_id: int) -> int:
        return self.conn.execute(r'''
            SELECT current_reply_count FROM thread
            WHERE id = ?
        ''', (thread_id,)).fetchone()[0]

    def is_thread_recorded(self, thread_id: int) -> bool:
        return self.conn.execute(r'''
            SELECT count(id) FROM thread WHERE id = ?
        ''', (thread_id,)).fetchone()[0] == 1

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

    def get_thread_ids_seen_after(self, datetime: datetime) -> List[int]:
        """
        获取曾在指定时间后看到过的各串的串号
        """
        rows = self.conn.execute(r'''
            SELECT DISTINCT thread.id
            FROM thread
            JOIN post ON thread.id = post.parent_thread_id
            WHERE thread.created_at >= ? OR post.created_at >= ?
        ''', (datetime.timestamp(), datetime.timestamp())).fetchall()

        return list(map(lambda x: x[0], rows))

    def is_thread_disappeared(self, thread_id: int) -> bool:
        return (self.conn.execute(r'''
            SELECT is_disappeared FROM thread_extra
            WHERE id = ?
        ''', (thread_id,)).fetchone() or [False])[0]

    def report_is_thread_disappeared(self, thread_id: int, checked_at: datetime, value: bool):
        self.conn.execute(r'''
            INSERT INTO thread_extra (id, checked_at, is_disappeared)
                VALUES (?, ?, ?)
            ON CONFLICT (id) DO
                UPDATE SET checked_at = ?, is_disappeared = ?
        ''', (thread_id, checked_at.timestamp(), value, checked_at.timestamp(), value))
