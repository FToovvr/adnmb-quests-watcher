#!/usr/bin/env python3

from __future__ import annotations
from typing import Tuple, List, Dict, OrderedDict, Optional
from dataclasses import dataclass

import os
import sqlite3
from datetime import datetime, timedelta, time
import logging
import logging.config
import traceback
import re

from dateutil import tz
from bs4 import BeautifulSoup
import requests

import anobbsclient
from anobbsclient.walk import create_walker, ReversalThreadWalkTarget

from commons import client, Trace

TREND_THREAD_ID = int(os.environ['ANOBBS_QUESTS_TREND_THREAD_ID'])

local_tz = tz.gettz('Asia/Shanghai')

RANK_LIMIT = 32

MAIN_DIVIDER_PART = "══" + '\u200b' + "══" + '\u200b' + "══"
META_MAIN_DIVIDER = f"{MAIN_DIVIDER_PART}　META　{MAIN_DIVIDER_PART}"

DEBUG_JUST_PRINT_REPORT = False


def main():

    now = datetime.now(tz=local_tz)
    if now.time() < time(hour=4):
        now -= timedelta(hours=5)
    yesterday = now - timedelta(days=1)

    if DEBUG_JUST_PRINT_REPORT:
        (_, _, content) = retrieve_data_then_generate_trend_report_text(yesterday, None)
        print(content)

    trace = Trace(conn=sqlite3.connect('db.sqlite3'), date=now.date())
    attempts = trace.attempts
    if attempts > 3:
        return
    trace.attempts = attempts + 1

    if trace.is_done:
        return

    logging.config.fileConfig('logging.3_publish_report.conf')

    logging.info(f"开始进行发布报告相关流程。UUID={trace.uuid}")

    if not trace.has_made_reply_request:
        logging.info("尚未发送回复请求以发布报告，将生成报告文本并尝试发送回复请求")

        (title, name, content) = \
            retrieve_data_then_generate_trend_report_text(
                yesterday, trace.uuid)

        try:
            client.reply_thread(content, to_thread_id=TREND_THREAD_ID,
                                title=title, name=name)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logging.warning("请求服务器超时，将尝试检查是否成功发串")
            logging.warning(traceback.format_exc())
        except anobbsclient.ReplyException as e:
            logging.critical(
                f"服务器响应表示发布回应失败：error={e.raw_error}；detail={e.raw_detail}")
            raise e
        except Exception as e:
            logging.critical(f"发布回应失败：{e}")
            logging.critical(traceback.format_exc())
            raise e
        else:
            logging.info("服务器响应表示发布回应成功")
            trace.report_made_reply_request()
    else:
        logging.info("早前已发送回复请求以发布报告，将进行下一步")

    assert(trace.reply_post_id is None)
    logging.info("尚未有查找到回应的记录，将查找之")

    post = find_last_post_with_uuid(TREND_THREAD_ID)
    if post is None:
        logging.error("未找到任何带有 UUID 回应，本次终止")
        exit(1)

    (post_id, uuid, offset) = post
    if uuid != trace.uuid:
        logging.error("最后带有 UUID 的回应与本次的不匹配，本次终止。找到的 UUID={uuid}")
        exit(1)

    logging.info(f"找到所求回应，将记录。回应串号={post_id}，偏移={offset}")

    trace.report_found_reply_post(thread_id=TREND_THREAD_ID,
                                  post_id=post_id, offset=offset)

    logging.info("成功结束")


def find_last_post_with_uuid(thread_id: int) -> Optional[Tuple[int, str, int]]:
    """
    Returns
    -------
    [0] : int
        目标回应的串号
    [1] : str
        找到的 UUID
    [2] : int
        目标回应的偏移

    如果没找到或者找到的第一个 uuid 不匹配，返回 None
    """

    # TODO: 可以根据上一次回应所在位置预测一下，大部分情况能把请求减少到1次
    # TODO: 如果发现串 SAGE 了，以后就不发了，或者提前检查一下有无 SAGE？

    (page_1, _) = client.get_thread_page(
        id=TREND_THREAD_ID, page=1, for_analysis=1)
    page_1: anobbsclient.ThreadPage = page_1
    # TODO: 其实这个可以在 API 那边定义 property 来算吧
    total_pages = (page_1.body.total_reply_count - 1) // 19 + 1

    walker = create_walker(
        target=ReversalThreadWalkTarget(
            thread_id=TREND_THREAD_ID,
            gatekeeper_post_id=None,
            start_page_number=total_pages,
        ),
        client=client,
    )

    for (pn, page, _) in walker:
        page: anobbsclient.ThreadPage = page
        for (i, post) in enumerate(reversed(page.replies)):
            text = BeautifulSoup(post.content).text
            uuid_rx = re.compile(
                r"(?:.*\n)+" + META_MAIN_DIVIDER + r"\n" +
                r"(?:.*\n)+UUID=([0-9a-f\-]+).*(?:\n.*)*",
                re.MULTILINE,
            )
            result = uuid_rx.match(text)
            if result is not None:
                uuid = result.group(1)
                return (post.id, uuid, (pn-1)*19+1+(len(page.replies)-1-i))
    return None


def retrieve_data_then_generate_trend_report_text(date: datetime, uuid: str) -> Tuple[str, str, str]:
    db = DB(conn=sqlite3.connect('file:db.sqlite3?mode=ro', uri=True))
    threads = db.get_daily_threads(date)
    threads_one_day_before = db.get_daily_threads(
        date - timedelta(days=1))
    tail_frequencies = db.get_tail_frequencies(date)
    stats = db.get_meta_stats(date)
    db.close()

    return generate_trend_report_text(
        date, threads, threads_one_day_before, tail_frequencies, stats, uuid)


def generate_trend_report_text(date: datetime,
                               threads: List[ThreadStats],
                               threads_one_day_before: Optional[List[ThreadStats]],
                               tail_frequencies: Tuple[int, Dict[int, float]],
                               stats: Stats, uuid: str) -> Tuple[str, str, str]:
    """
    Returns
    -------
    [0] : str
        标题
    [1] : str
        名称
    [2] : str
        正文
    """

    title = date.strftime("日度趋势 %Y-%m-%d")
    name = "当日4AM～次日4AM前"

    # 由于引用中看不到标题和名称，将标题和名称额外放到正文中
    lines = [
        date.strftime("【跑团版 趋势 日度报告】〔%Y-%m-%d〕"),
        f"统计范围：当日上午4时～次日上午4时前",
        '',
    ]

    def format_counts(threads: List[ThreadStats],
                      threads_one_day_before: Optional[List[ThreadStats]]) -> str:

        def calculate_counts(threads: List[ThreadStats]) -> Tuple[int, int]:
            return (
                len(list(filter(lambda x: x.is_new, threads))),
                sum(list(map(lambda x: x.increased_response_count, threads))),
            )

        (new_threads, new_posts) = calculate_counts(threads)
        new_threads_before, new_posts_before = None, None
        if threads_one_day_before is not None:
            (new_threads_before, new_posts_before) = calculate_counts(
                threads_one_day_before)

        def format_delta(delta: Optional[int]) -> str:
            if delta is not None:
                if delta > 0:
                    return f"(↑{delta})"
                elif delta < 0:
                    return f"(↓{abs(delta)})"
                return f"→0"
            return ""

        return [
            f"新增主题串 {new_threads}{format_delta(new_threads-new_threads_before)} 串",
            f"新增回应 {new_posts}{format_delta(new_posts-new_posts_before)} 条",
        ]

    count_texts = format_counts(threads, threads_one_day_before)

    lines += [
        "统计范围内：",
        f"　∗ {count_texts[0]}",
        f"　∗ {count_texts[1]}",
    ]

    lines += ['', f"{MAIN_DIVIDER_PART}　趋势　{MAIN_DIVIDER_PART}"]

    for (i, thread) in enumerate(threads):
        rank = i + 1
        if rank > RANK_LIMIT:
            # 让并列的串也上榜
            if thread.increased_response_count != threads[i-1].increased_response_count:
                break

        lines += ['']

        head = f"#{('%02d' % rank)}"
        if thread.is_new:
            head += f" [+{thread.increased_response_count} 回应 NEW!]"
        else:
            head += f" [+{thread.increased_response_count} ={thread.total_reply_count} 回应]"
        head += f" [@{thread.created_at.strftime('%Y-%m-%d')}" \
            + f"{ thread.created_at.strftime(' %H:%M') if thread.is_new else ''}]"
        lines += [head]

        subhead = []
        subhead += [f"(+{thread.increased_response_count_by_po} PO回应)"]
        subhead += [f"({thread.distinct_cookie_count} 参与饼数)"]
        lines += ["　"*2 + " ".join(subhead)]

        lines += [f">>No.{thread.id}"]

        lines += thread.generate_summary(free_lines=3).split('\n')

        lines += [(
            "━━━━" + '\u200b' + "━━━━"
            + '\u200b'
            + "━━━━" + '\u200b' + "━━━━")]

    lines += ['', f"{MAIN_DIVIDER_PART}　其它　{MAIN_DIVIDER_PART}"]

    def format_tail_frequencies(count: int, tail_frequencies: OrderedDict[int, float]) -> str:
        text = f"「r」尾号出目频率：\n"
        tail_frequencies.move_to_end(0)
        f = list(reversed(tail_frequencies.items()))
        f = list(map(lambda x: "{}={:05.2f}%".format(*x), f))
        for i in range(0, 10, 4):
            text += ' '.join(f[i:i+4]) + '\n'
        return text

    lines += ['', format_tail_frequencies(tail_frequencies[0],
                                          tail_frequencies[1])]

    lines += ['', META_MAIN_DIVIDER]
    lines += ['',
              f"统计期间：共上传 {stats.total_bandwidth_usage[0]} 字节，下载 {stats.total_bandwidth_usage[1]} 字节。"]
    lines += ['', f"UUID={uuid} # 定位用"]

    return (title, name, '\n'.join(lines))


def generate_new_thread_report_text(date: datetime, threads: List[ThreadStats]):

    new_threads = list(filter(lambda x: x.is_new, threads))
    new_threads = sorted(new_threads, key=lambda x: x.id)

    lines = [
        date.strftime("日度报告 新串 %Y-%m-%d"),
        date.strftime("统计范围：当日上午4时～次日上午4时前"),
        "＞以下所列新串将按发布时间倒序排列＜"
    ]

    lines += ['', f"{MAIN_DIVIDER_PART}　新串　{MAIN_DIVIDER_PART}"]

    for thread in reversed(new_threads):
        # 由于新串发得越早，积累回应的时间便越多，越有优势，这里排列就按时间反着来

        lines += ['']

        lines += [
            "╭" + "┅" * 4
            + thread.created_at.strftime('%m-%d %H:%M')
            + "┅" * 5
        ]
        lines += [f">>No.{thread.id}"]
        lines += thread.generate_summary(free_lines=5).split('\n')

    return '\n'.join(lines)


@dataclass(frozen=True)
class ThreadStats:
    id: int
    created_at: datetime
    is_new: bool
    title: Optional[str]
    name: Optional[str]
    raw_content: str
    increased_response_count: int
    total_reply_count: int
    increased_response_count_by_po: int
    distinct_cookie_count: int

    @property
    def content(self) -> str:
        return BeautifulSoup(self.raw_content, features='html.parser').get_text()

    def generate_summary(self, free_lines: int) -> str:
        lines = []
        if self.title is not None:
            title = self.title.replace('\u200b', '')
            if len(title) > 15:  # 以防万一
                title = title[:14] + "…"
            free_lines -= 1
            lines += [f"标题：{title}"]
        if self.name is not None:
            name = self.name.replace('\u200b', '')
            if len(name) > 15:  # 以防万一
                name = name[:14] + "…"
            free_lines -= 1
            lines += [f"名称：{name}"]
        for content_line in self.content.split('\n'):
            if free_lines == 0:
                lines += ["…"]
                break
            content_line = content_line.rstrip()
            line_to_add = ""
            for line_part in [content_line[i: i+16] for i in range(0, len(content_line), 16)]:
                if free_lines == 0:
                    line_to_add += "…"
                    break
                line_to_add += line_part.replace('\u200b', '')
                free_lines -= 1
            lines += [line_to_add]
        while True:
            if lines[-1].strip() == "":
                lines.pop()
            else:
                break

        return "\n".join(lines)


@dataclass(frozen=True)
class Stats:
    total_bandwidth_usage: anobbsclient.BandwidthUsage


@dataclass(frozen=True)
class DB:

    conn: sqlite3.Connection

    def close(self):
        self.conn.close()

    def get_daily_threads(self, date: datetime) -> List[ThreadStats]:
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute('''
            WITH later_changes AS (
                SELECT
                    parent_thread_id AS thread_id,
                    count(post.id) AS increased_response_count
                FROM post
                LEFT JOIN thread ON post.parent_thread_id = thread.id
                WHERE post.created_at >= ?
                GROUP BY parent_thread_id
            )
            SELECT
                parent_thread_id,
                thread.created_at,
                thread.title,
                thread.name,
                thread.content,
                count(post.id) AS increased_response_count,
                thread.current_reply_count - COALESCE(later_changes.increased_response_count, 0),
                SUM(CASE WHEN post.user_id = thread.user_id THEN 1 ELSE 0 END),
                count(DISTINCT post.user_id)
            FROM post
            LEFT JOIN thread ON post.parent_thread_id = thread.id
            LEFT JOIN later_changes ON thread.id = later_changes.thread_id
            WHERE post.created_at >= ? and post.created_at < ?
            GROUP BY parent_thread_id
            ORDER BY increased_response_count DESC
        ''', (upper_bound.timestamp(), lower_bound.timestamp(), upper_bound.timestamp())).fetchall()

        threads: List[ThreadStats] = []
        for row in rows:
            created_at = datetime.fromtimestamp(row[1], tz=local_tz)
            threads.append(ThreadStats(
                id=row[0],
                created_at=created_at,
                is_new=created_at >= lower_bound and created_at < upper_bound,
                title=row[2],
                name=row[3],
                raw_content=row[4],
                increased_response_count=row[5],
                total_reply_count=row[6],
                increased_response_count_by_po=row[7],
                distinct_cookie_count=row[8],
            ))

        return threads

    def get_meta_stats(self, date: datetime) -> Stats:
        lower_bound, upper_bound = self._get_boundaries(date)

        row = self.conn.execute('''
        SELECT 
            sum(uploaded_bytes), sum(downloaded_bytes),
            sum(requested_board_page_count), sum(requested_thread_page_count)
        FROM activity
        WHERE fetched_since >= ? and fetched_since < ?
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()

        return Stats((row[0], row[1]))

    def get_tail_frequencies(self, date: datetime) -> Tuple[int, Dict[int, float]]:
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute('''
            WITH tail_count AS (
                WITH post_tail AS (
                    SELECT id % 10 AS tail_number
                    FROM post
                    WHERE created_at >= ? and created_at < ?
                        AND content LIKE 'r'
                )
                SELECT tail_number, count(tail_number) AS `count`
                FROM post_tail
                GROUP BY tail_number
            )
            SELECT tail_number, `count` * 1.0 / (SELECT sum(`count`) FROM tail_count) * 100 AS percentage
            FROM tail_count
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchall()

        frequencies = OrderedDict({r[0]: r[1] for r in rows})

        count = self.conn.execute('''
            SELECT count(id) FROM post
            WHERE created_at >= ? and created_at < ?
                AND content LIKE 'r'
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()[0]

        return (count, frequencies)

    def _get_boundaries(self, date: datetime) -> Tuple[datetime, datetime]:
        lower_bound = date.replace(hour=4, minute=0, second=0, microsecond=0)
        upper_bound = lower_bound + timedelta(days=1)
        return (lower_bound, upper_bound)


if __name__ == '__main__':
    main()
