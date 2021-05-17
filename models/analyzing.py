from typing import Optional, Union, Tuple, List, Dict, OrderedDict
from dataclasses import dataclass

from datetime import datetime, timedelta, date

from bs4 import BeautifulSoup
import psycopg2

import anobbsclient

import sys
sys.path.append("..")  # noqa

# pylint: disable=import-error
from commons.consts import local_tz
from commons.thread_stats import ThreadStats, Counts


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

    def get_responses_match(self, date: datetime, in_thread_id: int, content_pattern: str) -> List[Tuple[int, str, int]]:
        """
        获取该日某串中匹配所给正则表达式的那些回应。

        Returns
        -------
        [0] : int
            匹配回应的串号。
        [1] : str
            匹配回应的内容。
        [2] : int
            匹配回应的偏移。

            如果在匹配回应发布后到本次统计期间，有在此之后的回应被删，可能会前移。
        """
        lower_bound, upper_bound = self._get_boundaries(date)

        self.cur.execute(r'''SELECT * FROM get_responses_match(%s, %s, %s, %s)''',
                         (in_thread_id, content_pattern, lower_bound, upper_bound))
        return self.cur.fetchall()

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
