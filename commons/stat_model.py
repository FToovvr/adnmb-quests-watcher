from typing import Optional, Union, Tuple, List, Dict, OrderedDict
from dataclasses import dataclass

from datetime import datetime, timedelta, date
import sqlite3
import re
import statistics

from bs4 import BeautifulSoup, Tag

import anobbsclient

from .consts import local_tz, ZWSP, OMITTING


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

    @property
    def blue_text(self) -> Optional[str]:
        soup = BeautifulSoup(self.raw_content, features='html.parser')

        def find_fn(tag: Tag):
            if tag.name == 'font':
                return re.match(r'^\s*blue\s*$', tag.get('color', '')) is not None
            if tag.name == 'span':
                return re.match(r'^.*;?\s*color:\s*blue\s*;?.*$', tag.get('style', '')) is not None
            return False

        elems = soup.find_all(find_fn)
        if len(elems) == 0:
            return None
        return elems[-1].get_text()


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

        rows = self.conn.execute(r'''
            WITH later_changes AS (
                SELECT
                    parent_thread_id AS thread_id,
                    count(post.id) AS increased_response_count
                FROM post
                LEFT JOIN thread ON post.parent_thread_id = thread.id
                WHERE post.created_at >= ?
                GROUP BY parent_thread_id
            ), revision_at_that_time AS (
                WITH tmp AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY id
                        ORDER BY not_anymore_at_least_after ASC
                    ) AS row_number
                    FROM thread_old_revision
                    WHERE not_anymore_at_least_after >= ?
                )
                SELECT id, title, name, content
                FROM tmp
                WHERE row_number = 1
            )
            SELECT
                parent_thread_id,
                thread.created_at,
                CASE WHEN revision_at_that_time.id IS NULL THEN thread.title   ELSE revision_at_that_time.title   END,
                CASE WHEN revision_at_that_time.id IS NULL THEN thread.name    ELSE revision_at_that_time.name    END,
                CASE WHEN revision_at_that_time.id IS NULL THEN thread.content ELSE revision_at_that_time.content END,
                count(post.id) AS increased_response_count,
                thread.current_reply_count - COALESCE(later_changes.increased_response_count, 0),
                SUM(CASE WHEN post.user_id = thread.user_id THEN 1 ELSE 0 END),
                count(DISTINCT post.user_id)
            FROM post
            LEFT JOIN thread ON post.parent_thread_id = thread.id
            LEFT JOIN later_changes ON thread.id = later_changes.thread_id
            LEFT JOIN revision_at_that_time ON thread.id = revision_at_that_time.id
            WHERE post.created_at >= ? AND post.created_at < ?
            GROUP BY parent_thread_id
            ORDER BY increased_response_count DESC
        ''', (upper_bound.timestamp(), upper_bound.timestamp(), lower_bound.timestamp(), upper_bound.timestamp())).fetchall()

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

        row = self.conn.execute(r'''
            WITH daily_qst_replies_in_range AS (
                SELECT id, content
                FROM post
                WHERE parent_thread_id = ?
                    AND created_at >= ? AND created_at < ?
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
            daily_qst_thread_id,
            lower_bound.timestamp(), upper_bound.timestamp(),
            daily_qst_thread_id,
        )).fetchone()

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

    # 在 ``updating_model.DB`` 中重复存在
    # TODO: 也许该来个 DBBase？
    def is_thread_disappeared(self, thread_id: int) -> bool:
        return (self.conn.execute(r'''
            SELECT is_disappeared FROM thread_extra
            WHERE id = ?
        ''', (thread_id,)).fetchone() or [False])[0]
