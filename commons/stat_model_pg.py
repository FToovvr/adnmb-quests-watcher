from typing import Optional, Union, Tuple, List, Dict, OrderedDict
from dataclasses import dataclass

from datetime import datetime, timedelta, date
import sqlite3
import re
import statistics

from bs4 import BeautifulSoup, Tag
import psycopg2

import anobbsclient

from .consts import local_tz, ZWSP, OMITTING


@dataclass(frozen=True)
class ThreadStats:
    id: int
    created_at: datetime
    is_new: bool
    is_disappeared: bool

    title: Optional[str]
    name: Optional[str]
    raw_content: str

    total_reply_count: int
    increased_response_count: int
    increased_response_count_by_po: int
    distinct_cookie_count: int
    increased_character_count: int
    increased_character_count_by_po: int

    blue_text: Optional[str]
    are_blue_texts_new: bool

    @property
    def content(self) -> str:
        return BeautifulSoup(self.raw_content, features='html.parser').get_text()

    @staticmethod
    def insert_zwsps_everywhere(text: str) -> str:
        return ZWSP.join(list(text))

    def generate_summary(self, free_lines: int) -> str:
        # TODO: 其实插入 zwsp 放外部更合适？
        lines = []
        if self.title is not None:
            title = self.title.replace(ZWSP, '')
            if len(title) > 15:  # 以防万一
                title = title[:14] + OMITTING
            free_lines -= 1
            lines += [f"标题：{ThreadStats.insert_zwsps_everywhere(title)}"]
        if self.name is not None:
            name = self.name.replace(ZWSP, '')
            if len(name) > 15:  # 以防万一
                name = name[:14] + OMITTING
            free_lines -= 1
            lines += [f"名称：{ThreadStats.insert_zwsps_everywhere(name)}"]
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
            lines += [ThreadStats.insert_zwsps_everywhere(line_to_add)]
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

        if self.threads == 0:
            self.thread_new_post_average = 0  # 或者 None？
            self.thread_new_post_quartiles = [0] * 3
            self.thread_new_post_variance = 0
            return

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

    cur: psycopg2._psycopg.cursor
    conn: sqlite3.Connection

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

        self.cur.execute(r'''
            SELECT * FROM get_daily_threads_report(%s, %s)
        ''', (lower_bound, upper_bound)
        )
        rows = self.cur.fetchall()

        threads: List[ThreadStats] = []
        for row in rows:
            threads.append(ThreadStats(
                id=row[0],
                created_at=row[1],
                is_new=row[2],
                is_disappeared=row[3],

                title=row[4],
                name=row[5],
                raw_content=row[6],

                total_reply_count=row[7],
                increased_response_count=row[8],
                increased_response_count_by_po=row[9],
                distinct_cookie_count=row[10],
                increased_character_count=row[11],
                increased_character_count_by_po=row[12],

                blue_text=row[13],
                are_blue_texts_new=row[14],
            ))

        return threads

    def get_daily_qst(self, date: datetime, daily_qst_thread_id: int) -> Optional[Tuple[int, int]]:
        """
        获取该日的跑团日报。

        Returns
        -------
        [0] : int
            该日日报的串号。
        [1] : int
            该日日报的偏移。

            如果在日报发布后到本次统计期间，有在此之后的回应被删，可能会前移。
        """
        lower_bound, upper_bound = self._get_boundaries(date)

        self.cur.execute(r'''SELECT * FROM get_daily_qst_id_and_position(%s, %s, %s)''',
                         (daily_qst_thread_id, lower_bound, upper_bound))
        row = self.cur.fetchone()

        return None if row is None else (row[0], row[1])

    def get_meta_stats(self, date: datetime) -> Stats:
        lower_bound, upper_bound = self._get_boundaries(date)

        row = self.conn.execute(r'''
        SELECT 
            IFNULL(sum(uploaded_bytes), 0), IFNULL(sum(downloaded_bytes), 0),
            IFNULL(sum(requested_board_page_count), 0), IFNULL(sum(requested_thread_page_count), 0)
        FROM activity
        WHERE fetched_since >= ? AND fetched_since < ?
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()

        return Stats((row[0], row[1]))

    def get_tail_frequencies(self, date: datetime) -> Tuple[int, Dict[int, float]]:
        lower_bound, upper_bound = self._get_boundaries(date)

        rows = self.conn.execute(r'''
            WITH tail_count AS (
                WITH post_tail AS (
                    SELECT id % 10 AS tail_number
                    FROM post
                    WHERE created_at >= ? AND created_at < ?
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
            WHERE created_at >= ? AND created_at < ?
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
                WHERE created_at >= ? AND created_at < ?
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

    def _get_boundaries(self, date: date) -> Tuple[datetime, datetime]:
        lower_bound = datetime.fromisoformat(
            f"{date.isoformat()} 04:00:00").replace(tzinfo=local_tz)
        upper_bound = lower_bound + timedelta(days=1)
        return (lower_bound, upper_bound)
