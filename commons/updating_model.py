from typing import Any, Optional,  List, Dict
from dataclasses import dataclass, field

import sqlite3
import logging
from datetime import datetime, timedelta
import json
import traceback

import anobbsclient

from commons import local_tz


@dataclass
class TotalBandwidthUsage:
    usages: List[anobbsclient.BandwidthUsage] = field(default_factory=list)

    def add(self, new_usage: anobbsclient.BandwidthUsage):
        self.usages.append(new_usage)

    @property
    def total(self) -> anobbsclient.BandwidthUsage:
        return tuple(map(sum, zip(*self.usages)))


@dataclass
class Stats:

    # TODO: 这些是不是该在 DB 那边统计？
    new_thread_count = 0
    affected_thread_count = 0
    new_post_count = 0

    # TODO: 这些是不是该在 API 那边统计？
    board_request_count = 0
    thread_request_count = 0
    logged_in_thread_request_count = 0

    total_bandwidth_usage: TotalBandwidthUsage = field(
        default_factory=TotalBandwidthUsage)


@dataclass(frozen=True)
class DB:
    """


    TODO: 确保 actor?
    """

    conn: sqlite3.Connection

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger('DB'))

    activity_id: int = field(init=False)

    def __post_init__(self):
        self.conn.isolation_level = None  # 禁用自动提交

        cur = self.conn.cursor()
        cur.execute(r'''
            INSERT INTO activity (run_at, is_successful)
            VALUES (?, ?)
        ''', (datetime.now().timestamp(), False))
        activity_id = cur.execute(r'''
            SELECT id FROM activity WHERE rowid = ?
        ''', (cur.lastrowid,)).fetchone()[0]
        cur.close()
        object.__setattr__(self, 'activity_id', activity_id)

        self.logger.info(f'已开始新活动。活动 id = {activity_id}')

    @property
    def never_runs(self) -> bool:
        """返回是否从未运行过一次。"""
        never_runs = self.conn.execute(r'''
            SELECT count(id) = 0 FROM activity 
            WHERE is_successful = true AND ensured_fetched_until IS NOT NULL
            ''').fetchone()[0]
        return never_runs == 1

    @property
    def should_fetch_since(self) -> datetime:
        should_fetch_since = self.__should_fetch_since
        self.logger.info(
            f'正在汇报本次活动计划的抓取时间范围下限。活动 ID = {self.activity_id}，'
            + f'此下限 = {should_fetch_since}')

        self.conn.execute(r'''
            UPDATE activity SET fetched_since = ?
            WHERE id = ?
        ''', (should_fetch_since.timestamp(), self.activity_id))
        self.conn.commit()

        self.logger.info(f'已汇报本次活动计划的抓取时间范围下限。活动 ID = {self.activity_id}')
        return should_fetch_since

    @property
    def __should_fetch_since(self) -> datetime:
        """
        本次应该抓取的串/回应的发布时间应该不晚于此。

        不增加1秒是因为那样在极端情况下可能会有遗漏。
        """
        if self.never_runs:
            # 若第一次运行，以5分钟前作为抓取的时间下界
            return datetime.now().replace(tzinfo=local_tz) - timedelta(minutes=5)

        last_activity_fetched_until = self.conn.execute(r'''
            SELECT ensured_fetched_until FROM activity 
            WHERE is_successful = TRUE AND ensured_fetched_until IS NOT NULL
            ORDER BY ensured_fetched_until DESC LIMIT 1
            ''').fetchone()[0]
        return datetime.fromtimestamp(last_activity_fetched_until).replace(tzinfo=local_tz)

    def report_ensured_fetched_until(self, ensured_fetched_until: datetime):
        self.logger.info(
            f'正在汇报本次活动可确保的抓取时间范围上限。活动 ID = {self.activity_id}，'
            + f'此上限 = {ensured_fetched_until}')

        self.conn.execute(r'''
            UPDATE activity SET ensured_fetched_until = ?
            WHERE id = ?
        ''', (ensured_fetched_until.timestamp(), self.activity_id))
        self.conn.commit()

        self.logger.info(f'已汇报本次活动可确保的抓取时间范围上限。活动 ID = {self.activity_id}')

    def try_find_thread_latest_seen_reply_id(self, thread_id: int) -> Optional[int]:
        row = self.conn.execute(r'''
            SELECT id FROM post WHERE parent_thread_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (thread_id,)).fetchone()
        if row is None:
            return None
        return row[0]

    def record_thread(self, thread: anobbsclient.BoardThread, record_total_reply_count: bool = True):
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
        self.conn.commit()

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
        if len(post_raw) == 0:
            return None
        return json.dumps(post_raw)

    def report_end(self, exception: Exception, stats: Stats):
        if exception is not None:
            is_successful = False
            try:
                raise exception
            except Exception:
                message = traceback.format_exc()
        else:
            is_successful = True
            message = None
        total_usage = stats.total_bandwidth_usage.total

        self.logger.info(
            f'正在汇报本次活动结果。活动 ID = {self.activity_id}，成功 = {is_successful}，'
            + f'上传字节数 = {total_usage[0]}，下载字节数 = {total_usage[1]}，'
            + f'新记录串数 = {stats.new_thread_count}，有新增回应串数 = {stats.affected_thread_count}，'
            + f'新记录回应数 = {stats.new_post_count}，'
            + f'请求版块页面次数 = {stats.board_request_count}，请求串页面次数 = {stats.thread_request_count}，'
            + f'以登录状态请求串页面次数 = {stats.logged_in_thread_request_count}')

        self.conn.execute(r'''
            UPDATE activity
            SET is_successful = ?, message = ?,
                uploaded_bytes = ?, downloaded_bytes = ?,
                newly_recorded_thread_count = ?, affected_thread_count = ?,
                newly_recorded_post_count = ?,
                requested_board_page_count = ?, requested_thread_page_count = ?,
                logged_in_thread_request_count = ?
            WHERE id = ?
        ''', (is_successful, message,
              total_usage[0], total_usage[1],
              stats.new_thread_count, stats.affected_thread_count,
              stats.new_post_count,
              stats.board_request_count, stats.thread_request_count,
              stats.logged_in_thread_request_count,
              self.activity_id))
        self.conn.commit()

        self.logger.info(
            f'已汇报本次活动结果。活动 ID = {self.activity_id}')

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
        self.conn.commit()
