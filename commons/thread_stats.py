from typing import Optional, Union, List
from dataclasses import dataclass

from datetime import datetime
import statistics

from bs4 import BeautifulSoup
import regex

from .consts import ZWSP, WORD_JOINER, OMITTING


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
            if len(lines) == 0:
                break

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
