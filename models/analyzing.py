from typing import Optional, Union, Tuple, List, Dict, OrderedDict
from dataclasses import dataclass

from datetime import datetime, timedelta, date
import statistics

from bs4 import BeautifulSoup, Tag
import psycopg2
import regex

import anobbsclient

import sys
sys.path.append("..")  # noqa

# pylint: disable=import-error
from commons.consts import local_tz, ZWSP, WORD_JOINER, OMITTING


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
    def make_text_unsearchable(text: str) -> str:
        """在文本中间插入空格，以防止报告内容污染搜索结果"""

        def insert_zwsps_fn(match_obj):
            return ZWSP.join(list(match_obj.group(0)))
        text = regex.sub(r'\p{han}+', insert_zwsps_fn, text)

        def insert_word_joiner_fn(match_obj):
            return WORD_JOINER.join(list(match_obj.group(0)))
        text = regex.sub(r'\p{latin}+', insert_word_joiner_fn, text)
        return text

    def generate_summary(self, free_lines: int) -> str:
        # TODO: 其实插入 zwsp 放外部更合适？
        lines = []
        if self.title is not None and len(self.title) > 0:
            title = self.title.replace(ZWSP, '')
            if len(title) > 15:  # 以防万一
                title = title[:14] + OMITTING
            free_lines -= 1
            lines += [
                f"标题：{ThreadStats.make_text_unsearchable(title)}"]
        if self.name is not None and len(self.name) > 0:
            name = self.name.replace(ZWSP, '')
            if len(name) > 15:  # 以防万一
                name = name[:14] + OMITTING
            free_lines -= 1
            lines += [
                f"名称：{ThreadStats.make_text_unsearchable(name)}"]
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
            lines += [
                ThreadStats.make_text_unsearchable(line_to_add)]
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

    @staticmethod
    def format_blue_texts(blue_texts: Optional[str]):
        if blue_texts is None:
            return None
        return BeautifulSoup(blue_texts[0], 'html.parser').get_text()

    def get_daily_threads(self, date: datetime) -> List[ThreadStats]:
        lower_bound, upper_bound = self._get_boundaries(date)

        self.cur.execute(r'''SELECT * FROM get_daily_threads_report(%s, %s)''',
                         (lower_bound, upper_bound))
        rows = self.cur.fetchall()

        threads: List[ThreadStats] = []
        for [
            id, parent_board_id,  # TODO: parent_board_id 暂时还不知道要在哪筛选，毕竟现在只有 111（跑团版）一个值
            created_at, is_new, is_disappeared,

            title, name, content,

            total_response_count,
            increased_response_count,
            increased_response_count_by_po,
            distinct_cookie_count,
            increased_character_count,
            increased_character_count_by_po,

            blue_texts,
            are_blue_texts_new,
        ] in rows:
            threads.append(ThreadStats(
                id=id,
                created_at=created_at,
                is_new=is_new,
                is_disappeared=is_disappeared,

                title=title,
                name=name,
                raw_content=content,

                total_reply_count=total_response_count,
                increased_response_count=increased_response_count,
                increased_response_count_by_po=increased_response_count_by_po,
                distinct_cookie_count=distinct_cookie_count,
                increased_character_count=increased_character_count,
                increased_character_count_by_po=increased_character_count_by_po,

                blue_text=DB.format_blue_texts(blue_texts),
                are_blue_texts_new=are_blue_texts_new,
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

        self.cur.execute(r'''SELECT * FROM get_meta_stats(%s, %s)''',
                         (lower_bound, upper_bound))
        row = self.cur.fetchone()

        return Stats((row[0], row[1]))

    def get_tail_frequencies(self, date: datetime) -> Tuple[int, Dict[int, float]]:
        # TODO: 其实没必要在这里保持有序

        lower_bound, upper_bound = self._get_boundaries(date)

        self.cur.execute(r'''SELECT * FROM get_tail_count(%s, %s)''',
                         (lower_bound, upper_bound))
        rows = self.cur.fetchall()

        counts = OrderedDict({r[0]: r[1] for r in rows})
        sum_count = sum(counts.values())
        frequencies = OrderedDict((tail, float(count) / sum_count)
                                  for tail, count in counts.items())

        return (sum_count, frequencies)

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

        TODO
        ----
        其实没必要在这里保持有序。
        """
        lower_bound, upper_bound = self._get_boundaries(date)

        self.cur.execute(r'''SELECT * FROM get_count_of_tail_numbers_with_consecutive_digits(%s, %s, %s)''',
                         (n, lower_bound, upper_bound))
        return self.cur.fetchall()

    def _get_boundaries(self, date: date) -> Tuple[datetime, datetime]:
        lower_bound = datetime.fromisoformat(
            f"{date.isoformat()} 04:00:00").replace(tzinfo=local_tz)
        upper_bound = lower_bound + timedelta(days=1)
        return (lower_bound, upper_bound)
