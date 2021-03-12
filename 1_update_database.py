#!/usr/bin/env python3

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import traceback

import os
import sqlite3
from datetime import datetime, timedelta
import json
import logging
import logging.config

from dateutil import tz

import anobbsclient
from anobbsclient.walk import create_walker, BoardWalkTarget, ReversalThreadWalkTarget

# 默认单线程

logging.config.fileConfig('logging.1_update_database.conf')

BOARD_ID = 111  # 跑团版
local_tz = tz.gettz('Asia/Shanghai')

client = anobbsclient.Client(
    user_agent=os.environ['ANOBBS_CLIENT_ENVIRON'],
    host=os.environ['ANOBBS_HOST'],
    appid=os.environ.get('ANOBBS_CLIENT_APPID', None),
    default_request_options={
        'user_cookie': anobbsclient.UserCookie(userhash=os.environ['ANOBBS_USERHASH']),
        'login_policy': 'when_required',
        'gatekeeper_page_number': 100,
    },
)


def main():
    db = DB(conn=sqlite3.connect('db.sqlite3'))

    stats = Stats()

    fetching_since = db.should_fetch_since

    exception = None
    try:
        fetch_board(db, fetching_since=fetching_since,
                    stats=stats)
    except Exception as e:
        exception = e

    db.report_end(exception, stats)

    db.close()

    if exception is not None:
        raise exception


def fetch_board(db: DB, fetching_since: datetime, stats: Stats):

    logger = logging.getLogger('FETCH')

    walker = create_walker(
        target=BoardWalkTarget(
            board_id=BOARD_ID,
            start_page_number=1,
            stop_before_datetime=fetching_since,
        ),
        client=client,
    )
    is_first_found_thread = True
    threads_on_board: List[anobbsclient.BoardThread] = []
    bandwidth_usage_for_board = TotalBandwidthUsage()
    for (pn, page, usage) in walker:
        logger.info(f'获取到版块第 {pn} 页。纳入串数 = {len(page)}')
        bandwidth_usage_for_board.add(usage)
        stats.board_request_count += 1
        threads_on_board += page
    stats.total_bandwidth_usage.add(bandwidth_usage_for_board.total)
    logger.info(f'完成获取版块。总共纳入串数 = {len(threads_on_board)}，'
                + f'期间 (上传字节数, 下载字节数) = {bandwidth_usage_for_board.total}')

    for (i, thread) in enumerate(threads_on_board):
        logger.debug(f'串 #{i}。串号 = {thread.id}，'
                     + f'最后修改时间 = {thread.last_modified_time}')

        if is_first_found_thread:
            is_first_found_thread = False
            db.report_ensured_fetched_until(thread.last_modified_time)

        is_thread_recorded = db.is_thread_recorded(thread.id)
        if not is_thread_recorded:
            stats.new_thread_count += 1
        # 记录或更新串
        # current_reply_count 在后面一同记录
        db.record_thread(thread)

        if len(thread.replies) == 0:
            assert(thread.total_reply_count == 0)
            logger.debug(f'串 #{i} 暂无回应，到此结束')
            continue

        # 根据数据库中是否已存在该串之前抓取到的回应，
        # 来决定如何判断某回应是否是抓取目标
        latest_seen_reply_id = \
            db.try_find_thread_latest_seen_reply_id(thread_id=thread.id)
        has_old_records = latest_seen_reply_id is not None
        if has_old_records:
            def is_target(x): return x.id > latest_seen_reply_id
            logger.debug(f'串 #{i} 是之前已经抓取过的串，'
                         + f'将会通过之前抓取到的最大串号作为范围的下界')
        else:
            def is_target(x): return x.created_at >= fetching_since
            logger.debug(f'串 #{i} 是之前曾未抓取过的串，'
                         + f'将会通过规定的下界时间作为范围的下界')

        if thread.total_reply_count <= 5 \
                or not is_target(thread.replies[0]):
            # 要抓取的内容全在预览里，不用再进串里去翻了
            # TODO 判断是否没有剩余回应（len(thread.total_reply_count) <= 5）应该在 API 那边进行
            targets = list(
                [post for post in thread.replies if is_target(post)])
            if len(targets) > 0:
                if is_thread_recorded:
                    stats.affected_thread_count += 1
                stats.new_post_count += len(targets)
            db.record_thread_replies(
                thread=thread, replies=targets, total_reply_count=thread.total_reply_count)
            logger.debug(f'串 #{i} 由于全部需要抓取的回应已在预览之中，记录后到此结束。')
        else:
            # 反向遍历
            start_page_number = (thread.total_reply_count - 1) // 19 + 1
            logger.debug(f'串 #{i} 需要进入以抓取目标范围内的回应。' +
                         f'从回应总数推测出的当前页数 = {start_page_number}')
            if (thread.total_reply_count % 19) <= 5:
                # 最新一页的内容已经全部包含在预览中了，因此略过
                logger.debug(f'串 #{i} 由于最新一页的回应已全部包含在预览中，抓取时会略过该页')
                start_page_number -= 1

            needs_gatekeeper_post_id = False
            if has_old_records:
                last_reply_count = \
                    db.get_thread_total_reply_count(thread_id=thread.id)
                if last_reply_count is not None:
                    last_page_count = (last_reply_count - 1) // 19 + 1
                else:
                    last_page_count = None
                    logger.warning(f'串 #{i} 存在曾抓取到的回应，但却没有记录回应总数')
                if (last_page_count is None or not client.thread_page_requires_login(last_page_count)) \
                        and client.thread_page_requires_login(start_page_number):
                    needs_gatekeeper_post_id = True
                    logger.debug(f'串 #{i} 由于要抓取的内容需要登录，'
                                 + f'而之前抓取到的内容在需要登录之前，无法用以判断是否卡页，'
                                 + f'因而需要额外获取第 100 页来确认守门串号')
            elif client.thread_page_requires_login(start_page_number):
                needs_gatekeeper_post_id = True
                logger.debug(f'串 #{i} 由于要抓取的内容需要登录，'
                             + f'而之前曾未抓取过内容，无法用以判断是否卡页，'
                             + f'因而需要额外获取第 100 页来确认守门串号')

            if needs_gatekeeper_post_id:
                # TODO: 这一块应该放在 API 那边
                (gatekeeper_page, usage) = client.get_thread_page(
                    id=thread.id, page=client.get_thread_gatekeeper_page_number())
                stats.total_bandwidth_usage.add(usage)
                stats.thread_request_count += 1
                gatekeeper_post_id = gatekeeper_page.replies[-1].id
                logger.debug(f'串 #{i} 确认守门串号。守门串号 = {gatekeeper_post_id}')
            else:
                gatekeeper_post_id = None

            if has_old_records:
                walker = create_walker(
                    target=ReversalThreadWalkTarget(
                        thread_id=thread.id,
                        start_page_number=start_page_number,
                        gatekeeper_post_id=gatekeeper_post_id,
                        stop_before_post_id=latest_seen_reply_id,
                        expected_stop_page_number=last_page_count,
                    ),
                    client=client,
                )
            else:
                walker = create_walker(
                    target=ReversalThreadWalkTarget(
                        thread_id=thread.id,
                        start_page_number=start_page_number,
                        gatekeeper_post_id=gatekeeper_post_id,
                        stop_before_datetime=fetching_since,
                    ),
                    client=client,
                )

            final_reply_count = None
            targets = []
            bandwidth_usage_for_thread = TotalBandwidthUsage()
            thread_walk_page_count = 0
            for (pn, page, usage) in walker:
                thread_walk_page_count += 1
                stats.thread_request_count += 1
                if client.thread_page_requires_login(pn):
                    stats.logged_in_thread_request_count += 1
                logger.debug(f'串 #{i} 页 {pn}。纳入回应数 = {len(page.replies)}')
                page: anobbsclient.ThreadPage = page
                bandwidth_usage_for_thread.add(usage)
                if final_reply_count is None:
                    final_reply_count = page.body.total_reply_count
                targets += page.replies
            db.record_thread_replies(
                thread=thread, replies=targets, total_reply_count=final_reply_count)
            stats.total_bandwidth_usage.add(bandwidth_usage_for_thread.total)
            if len(targets) > 0:
                if is_thread_recorded:
                    stats.affected_thread_count += 1
                stats.new_post_count += len(targets)
            logger.debug(f'串 #{i} 已抓取到范围内所有新回应，记录后到此结束。'
                         + f'遍历访问页数 = {thread_walk_page_count}，'
                         + f'期间 (上传字节数, 下载字节数) = {bandwidth_usage_for_thread.total}')


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
        cur.execute('''
            INSERT INTO activity (run_at, is_successful)
            VALUES (?, ?)
        ''', (datetime.now().timestamp(), False))
        activity_id = cur.execute('''
            SELECT id FROM activity WHERE rowid = ?
        ''', (cur.lastrowid,)).fetchone()[0]
        cur.close()
        object.__setattr__(self, 'activity_id', activity_id)

        self.logger.info(f'已开始新活动。活动 id = {activity_id}')

    def close(self):
        self.logger.info('正在关闭 DB')
        self.conn.close()
        self.logger.info('已关闭 DB')

    @property
    def never_runs(self) -> bool:
        """返回是否从未运行过一次。"""
        never_runs = self.conn.execute('''
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

        self.conn.execute('''
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

        last_activity_fetched_until = self.conn.execute('''
            SELECT ensured_fetched_until FROM activity 
            WHERE is_successful = true AND ensured_fetched_until IS NOT NULL
            ORDER BY ensured_fetched_until DESC LIMIT 1
            ''').fetchone()[0]
        return datetime.fromtimestamp(last_activity_fetched_until).replace(tzinfo=local_tz)

    def report_ensured_fetched_until(self, ensured_fetched_until: datetime):
        self.logger.info(
            f'正在汇报本次活动可确保的抓取时间范围上限。活动 ID = {self.activity_id}，'
            + f'此上限 = {ensured_fetched_until}')

        self.conn.execute('''
            UPDATE activity SET ensured_fetched_until = ?
            WHERE id = ?
        ''', (ensured_fetched_until.timestamp(), self.activity_id))
        self.conn.commit()

        self.logger.info(f'已汇报本次活动可确保的抓取时间范围上限。活动 ID = {self.activity_id}')

    def try_find_thread_latest_seen_reply_id(self, thread_id: int) -> Optional[int]:
        row = self.conn.execute('''
            SELECT id FROM post WHERE parent_thread_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (thread_id,)).fetchone()
        if row is None:
            return None
        return row[0]

    def record_thread(self, thread: anobbsclient.BoardThread):
        self.logger.info(f'正在记录/更新串信息。串号 = {thread.id}')

        # 这样暴力地 replace 就成了，反正每次抓取都会抓取到全部的 fields
        # 不在这里记录 current_reply_count, 因为为了准确需要在抓取完其中的
        self.conn.execute('''
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
        cur.execute('BEGIN')

        self.conn.execute('''
            UPDATE thread SET current_reply_count = ?
            WHERE id =?
        ''', (total_reply_count, thread.id))

        for post in replies:
            # 重复 insert post 是出现了 bug 的征兆
            self.conn.execute('''
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

        cur.execute('COMMIT')
        cur.close()

        self.logger.info(f'已记录串中新抓取到的回应。串号 = {thread.id}')

    def get_thread_total_reply_count(self, thread_id: int) -> int:
        return self.conn.execute('''
            SELECT current_reply_count FROM thread
            WHERE id = ?
        ''', (thread_id,)).fetchone()[0]

    def is_thread_recorded(self, thread_id: int) -> bool:
        return self.conn.execute('''
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
            + f'请求版块页面次数 = {stats.board_request_count}，请求串页面次数 = {stats.thread_request_count}')

        self.conn.execute('''
            UPDATE activity
            SET is_successful = ?, message = ?,
                uploaded_bytes = ?, downloaded_bytes = ?,
                newly_recorded_thread_count = ?, affected_thread_count = ?,
                newly_recorded_post_count = ?,
                requested_board_page_count = ?, requested_thread_page_count = ?
            WHERE id = ?
        ''', (is_successful, message,
              total_usage[0], total_usage[1],
              stats.new_thread_count, stats.affected_thread_count,
              stats.new_post_count,
              stats.board_request_count, stats.thread_request_count,
              self.activity_id))
        self.conn.commit()

        self.logger.info(
            f'已汇报本次活动结果。活动 ID = {self.activity_id}')


if __name__ == '__main__':
    main()
