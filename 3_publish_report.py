#!/usr/bin/env python3

from __future__ import annotations
from typing import Tuple, List, Dict, OrderedDict, Optional, Union
from dataclasses import dataclass

import os
import sqlite3
from datetime import datetime, timedelta, time
import logging
import logging.config
import traceback
import re
import statistics

from dateutil import tz
from bs4 import BeautifulSoup
import requests

import anobbsclient
from anobbsclient.walk import create_walker, ReversalThreadWalkTarget

from commons import client, Trace

TREND_THREAD_ID = int(os.environ['ANOBBS_QUESTS_TREND_THREAD_ID'])
DAILY_QST_THREAD_ID = int(os.environ['ANOBBS_QUESTS_DAILY_QST_THREAD_ID'])

local_tz = tz.gettz('Asia/Shanghai')

RANK_LIMIT = 32

ZWSP = '\u200b'

MAIN_DIVIDER_PART = f"══{ZWSP}══{ZWSP}══"
META_MAIN_DIVIDER = f"{MAIN_DIVIDER_PART}　META　{MAIN_DIVIDER_PART}"
OMITTING = ZWSP + "…"

DEBUG_JUST_PRINT_REPORT = False


def main():

    now = datetime.now(tz=local_tz)
    if now.time() < time(hour=4):
        now -= timedelta(hours=5)
    yesterday = now - timedelta(days=1)

    if DEBUG_JUST_PRINT_REPORT:
        (_, _, content) = retrieve_data_then_generate_trend_report_text(yesterday, None)
        print(content)
        return

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
        logging.error(f"最后带有 UUID 的回应与本次的不匹配，本次终止。找到的 UUID={uuid}")
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
        目标回应的串号。
    [1] : str
        找到的 UUID。
    [2] : int
        目标回应的偏移。

    如果没找到或者找到的第一个 uuid 不匹配，返回 None。
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
    post = generate_trend_report_text(date, db, uuid)
    db.close()
    return post


def generate_trend_report_text(date: datetime, db: DB, uuid: str) -> Tuple[str, str, str]:
    """
    Returns
    -------
    [0] : str
        标题。
    [1] : str
        名称。
    [2] : str
        正文。
    """

    title = date.strftime("日度趋势 %Y-%m-%d")
    name = "页 ❬1 / 1❭"

    # 由于引用中看不到标题和名称，将标题和名称额外放到正文中
    lines = [
        date.strftime(f"【 {ZWSP} 跑团版 趋势 日度报告〔%Y-%m-%d〕】"),
        f"统计范围：当日上午4时～次日上午4时前",
        "页 ❬1 / 1❭",
        '',
    ]

    daily_qst = db.get_daily_qst(date)
    if daily_qst is not None:
        lines += [f"当期跑团日报：>>No.{daily_qst[1]}（P{(daily_qst[2]-1)//19+1}）", '']

    threads = db.get_daily_threads(date)

    counts = Counts(threads)

    # TODO: 参数决定是否与前日比较
    class AttrsNone:
        def __getattribute__(self, _):
            return None
    counts_before = AttrsNone()
    counts_before = Counts(db.get_daily_threads(date - timedelta(days=1)))

    def format_counts(counts: Counts, counts_before: Counts) -> str:

        def format_value_with_delta(value: int, old_value: Optional[int]) -> str:
            if old_value is None:
                return str(value)
            delta = value - old_value
            if delta > 0:
                return f"{value}(↑{delta})"
            elif delta < 0:
                return f"{value}(↓{abs(delta)})"
            return f"{value}(→0)"

        def format_q(q: List[float], old_q: List[float]) -> str:
            if old_q is None:
                old_q = [None] * len(q)
            q_texts = [f"Q₁={format_value_with_delta(q[0], old_q[0])}"]
            q_texts += [f"中位数={format_value_with_delta(q[1], old_q[1])}"]
            q_texts += [f"Q₃={format_value_with_delta(q[2], old_q[2])}"]
            return ' '.join(q_texts)

        return [
            f"总计出现主题串 {format_value_with_delta(counts.threads, counts_before.threads)} 串",
            f"新增主题串 {format_value_with_delta(counts.new_threads, counts_before.new_threads)} 串",
            f"新增回应 {format_value_with_delta(counts.new_posts, counts_before.new_posts)} 条",
            f"主题串新增回应 {format_q(counts.thread_new_post_quartiles, counts_before.thread_new_post_quartiles)}"
            # 没太大意义…
            # f"平均主题串新增回应 {counts.thread_new_post_average} 条，"
            # + f"中位 {counts.thread_new_post_median} 条，"
            # + f"S²={counts.thread_new_post_variance}"
        ]

    count_texts = format_counts(counts, counts_before)

    lines += ["统计范围内："]
    lines += list(map(lambda x: f"{ZWSP} ∗ {x}", count_texts))

    lines += ['', f"{MAIN_DIVIDER_PART}　趋势　{MAIN_DIVIDER_PART}"]
    # lines += [f"#01 ⇒ #{RANK_LIMIT:02d}"]

    for (i, thread) in enumerate(threads):
        rank = i + 1
        if rank > RANK_LIMIT:
            # 让并列的串也上榜
            if thread.increased_response_count != threads[i-1].increased_response_count:
                break

        lines += ['']

        head = f"#{rank:02d}"
        if thread.is_new:
            head += f" [+{thread.increased_response_count} 回应 NEW!]"
        else:
            head += f" [+{thread.increased_response_count} ={thread.total_reply_count} 回应]"
        head += f" [@{thread.created_at.strftime('%Y-%m-%d')}" \
            + f"{ thread.created_at.strftime(' %H:%M') if thread.is_new else ''}]"
        lines += [head]

        subhead = []
        subhead += [f"(PO回应={thread.increased_response_count_by_po})"]
        approx_distinct_cookie_count = thread.distinct_cookie_count//5*5
        if approx_distinct_cookie_count != 0:
            subhead += [f"(参与饼干≥{approx_distinct_cookie_count})"]
        else:
            subhead += [f"(参与饼干>0)"]
        lines += ["　"*2 + " ".join(subhead)]

        lines += [f">>No.{thread.id}"]

        lines += thread.generate_summary(free_lines=3).split('\n')

        lines += [ZWSP.join([f"━━━━"]*4)]

    lines += ['', f"{MAIN_DIVIDER_PART}　其它　{MAIN_DIVIDER_PART}"]

    tail_frequencies = db.get_tail_frequencies(date)

    def format_tail_frequencies(count: int, tail_frequencies: OrderedDict[int, float]) -> str:
        text = f"此间，「r」串尾出目频率 (n={count})：\n"
        tail_frequencies.move_to_end(0)
        f = list(reversed(tail_frequencies.items()))
        f_max_min = [max(f, key=lambda x: x[1])[0],
                     min(f, key=lambda x: x[1])[0]]
        f = list(map(lambda x: (
            "{}={:05.2f}%*" if x[0] in f_max_min else "{}={:05.2f}% ").format(*x), f))
        for i in range(0, 10, 4):
            text += ' '.join(f[i:i+4]) + ("（*最高/最低）" if i == 8 else "") + '\n'
        return text

    lines += ['', format_tail_frequencies(tail_frequencies[0],
                                          tail_frequencies[1])]

    lucky_numbers = db.get_consecutive_tail_counts(date, 3)
    lines += ["此间，串尾连号次数："]
    lucky_lines = []
    for (n, count, zero_count) in lucky_numbers:
        text = "{} 连号 {} 次 ({:.2f}‰)，".format(
            n, count, count / counts.new_posts * 1000)
        if zero_count > 0:
            text += "其中全 0 有 {} 次 ({:.2f}‰)".format(
                zero_count, zero_count / counts.new_posts * 1000)
        else:
            text += "其中没有全 0"
        lucky_lines += [text]
    lines += ["；\n".join(lucky_lines) + "。"]

    lines += ['', META_MAIN_DIVIDER]

    stats = db.get_meta_stats(date)

    lines += ['',
              f"统计期间：共上传 {stats.total_bandwidth_usage[0]:,} 字节，下载 {stats.total_bandwidth_usage[1]:,} 字节。"]
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
            title = self.title.replace(ZWSP, '')
            if len(title) > 15:  # 以防万一
                title = title[:14] + OMITTING
            free_lines -= 1
            lines += [f"标题：{title}"]
        if self.name is not None:
            name = self.name.replace(ZWSP, '')
            if len(name) > 15:  # 以防万一
                name = name[:14] + OMITTING
            free_lines -= 1
            lines += [f"名称：{name}"]
        for content_line in self.content.split('\n'):
            if free_lines == 0:
                lines += [OMITTING]
                break
            content_line = content_line.rstrip()
            line_to_add = ""
            for line_part in [content_line[i: i+16] for i in range(0, len(content_line), 16)]:
                if free_lines == 0:
                    line_to_add += OMITTING
                    break
                line_to_add += line_part.replace(ZWSP, '')
                free_lines -= 1
            lines += [line_to_add]
        while True:
            if lines[-1].strip() == "":
                lines.pop()
            else:
                break

        return "\n".join(lines)


@dataclass  # (frozen=True)
class Counts:

    threads: int
    new_threads: int
    new_posts: int

    thread_new_post_average: int
    thread_new_post_quartiles: List[Union[float, int]]
    thread_new_post_variance: float

    def __init__(self, threads: List[ThreadStats]):
        self.threads = len(threads)
        self.new_threads = len(list(filter(lambda x: x.is_new, threads)))
        new_post_counts = list(
            map(lambda x: x.increased_response_count, threads))
        self.new_posts = sum(new_post_counts)

        self.thread_new_post_average = self.new_posts / self.threads
        q = statistics.quantiles(new_post_counts)
        q = list(map(lambda x: int(x) if x.is_integer else x, q))
        self.thread_new_post_quartiles = q
        self.thread_new_post_variance = statistics.variance(new_post_counts)


@dataclass(frozen=True)
class Stats:
    total_bandwidth_usage: anobbsclient.BandwidthUsage


@dataclass(frozen=True)
class DB:

    conn: sqlite3.Connection

    def close(self):
        self.conn.close()

    def __post_init__(self):

        # python 的 sqlite3 不支持 REGEXP
        def rx_test(pattern: str, string: str) -> bool:
            return re.search(pattern, string) is not None
        self.conn.create_function("rx_test", 2, rx_test)

        def rx_nth_match(pattern: str, string: str, nth: int) -> Optional[str]:
            r = re.match(pattern, string)
            return None if r is None else r.group(nth)
        self.conn.create_function("rx_nth_match", 3, rx_nth_match)

    def get_daily_threads(self, date: datetime) -> List[ThreadStats]:
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute(r'''
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

    def get_daily_qst(self, date: datetime) -> Optional[Tuple[int, int, int]]:
        """
        获取该日的跑团日报。

        Returns
        -------
        [0] : int
            日报串的串号。
        [1] : int
            该日日报的串号。
        [2] : int
            该日日报的偏移。

            如果在日报发布后到本次统计期间，有在此之后的回应被删，可能会前移。
        """
        lower_bound, upper_bound = self._get_boundaries(date)

        row = self.conn.execute(r'''
            WITH daily_qst_replies_in_range AS (
                SELECT id, content
                FROM post
                WHERE parent_thread_id = ?
                    AND created_at >= ? and created_at < ?
            )
            SELECT
                a.id, current_reply_count - count(b.id)
            FROM daily_qst_replies_in_range AS a
            JOIN daily_qst_replies_in_range AS b ON b.id > a.id
            LEFT JOIN thread ON thread.id = ?
            WHERE rx_test("(.*\n)*\[头条\]\s*(<br />)?(\n.*)+", a.content)
			GROUP BY a.id
			ORDER BY a.id DESC
			LIMIT 1
        ''', (
            DAILY_QST_THREAD_ID,
            lower_bound.timestamp(), upper_bound.timestamp(),
            DAILY_QST_THREAD_ID,
        )).fetchone()

        return None if row is None else (DAILY_QST_THREAD_ID, row[0], row[1])

    def get_meta_stats(self, date: datetime) -> Stats:
        lower_bound, upper_bound = self._get_boundaries(date)

        row = self.conn.execute(r'''
        SELECT 
            sum(uploaded_bytes), sum(downloaded_bytes),
            sum(requested_board_page_count), sum(requested_thread_page_count)
        FROM activity
        WHERE fetched_since >= ? and fetched_since < ?
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()

        return Stats((row[0], row[1]))

    def get_tail_frequencies(self, date: datetime) -> Tuple[int, Dict[int, float]]:
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute(r'''
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

        count = self.conn.execute(r'''
            SELECT count(id) FROM post
            WHERE created_at >= ? and created_at < ?
                AND content LIKE 'r'
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()[0]

        return (count, frequencies)

    def get_consecutive_tail_counts(self, date: datetime, n: int) -> Tuple[int, int, int]:
        """
        Returns
        -------
        [0]
            位数。
        [1]
            个数。
        [2]
            0 的个数。
        """
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute(r'''
        WITH lucky_tail_info AS (
            WITH lucky_tail AS (
                SELECT rx_nth_match("\d+?((\d)\2+)$", cast(id AS TEXT), 1) AS tail
                FROM post
                WHERE created_at >= ? and created_at < ?
                    AND rx_test("(\d)\1{" || ? || ",}$", cast(id AS TEXT))
                ORDER BY id
            )
            SELECT
                length(tail) AS length,
                cast(tail AS INTEGER) = 0 AS is_zero
            FROM lucky_tail
        )
        SELECT
            length,
            COUNT(*) AS `count`,
            SUM(is_zero) AS `zero_count`
        FROM lucky_tail_info
        GROUP BY length
        ORDER BY length DESC
        ''', (lower_bound.timestamp(), upper_bound.timestamp(), n-1))

        return rows.fetchall()

    def _get_boundaries(self, date: datetime) -> Tuple[datetime, datetime]:
        lower_bound = date.replace(hour=4, minute=0, second=0, microsecond=0)
        upper_bound = lower_bound + timedelta(days=1)
        return (lower_bound, upper_bound)


if __name__ == '__main__':
    main()
