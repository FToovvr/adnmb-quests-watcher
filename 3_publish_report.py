#!/usr/bin/env python3

from __future__ import annotations
from typing import Tuple, List, Dict, OrderedDict, Optional, Union
from dataclasses import dataclass, field

import os
import sqlite3
from datetime import datetime, timedelta, time
import logging
import logging.config
import traceback
import re
from time import sleep

import requests
from bs4 import BeautifulSoup

import anobbsclient
from anobbsclient.walk import create_walker, ReversalThreadWalkTarget

from commons import client, Trace, local_tz, ZWSP, OMITTING
from commons.stat_modal import ThreadStats, Counts, Stats, DB

TREND_THREAD_ID = int(os.environ['ANOBBS_QUESTS_TREND_THREAD_ID'])
DAILY_QST_THREAD_ID = int(os.environ['ANOBBS_QUESTS_DAILY_QST_THREAD_ID'])


RANK_LIMIT = 32

MAIN_DIVIDER_PART = f"══{ZWSP}══{ZWSP}══"
META_MAIN_DIVIDER = f"{MAIN_DIVIDER_PART}　META　{MAIN_DIVIDER_PART}"

DEBUG_JUST_PRINT_REPORT = False
DEBUG_NOTIFY_TO_TREND_THREAD = False


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

    # TODO: 检查成功与否
    # TODO: 开关决定是否通知
    notify_to_thread_id = DAILY_QST_THREAD_ID
    if DEBUG_NOTIFY_TO_TREND_THREAD:
        notify_to_thread_id = TREND_THREAD_ID
    logging.info(f"将发送报告出炉通知。由于发串间隔限制，将等待30秒")
    sleep(30)
    client.reply_thread(
        to_thread_id=notify_to_thread_id,
        title="本期跑团版趋势报告已出炉",
        name=yesterday.strftime("%Y-%m-%d"),
        content='\n'.join([
            yesterday.strftime("%Y年%-m月%-d日 跑团版 趋势日度报告："),
            f">>No.{post_id}（位于原串第{(offset-1) // 19 + 1}页）",
        ]),
    )

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
    with sqlite3.connect('file:db.sqlite3?mode=ro', uri=True) as conn:
        db = DB(conn=conn)
        return TrendReportTextGenerator(
            db=db,
            date=date,
            rank_limit=RANK_LIMIT,  # TODO: 允许由命令行参数改变
            uuid=uuid,
            should_compare_with_last_day=True,  # TODO: 同上
        ).generate()


@dataclass(frozen=True)
class TrendReportTextGenerator:

    db: DB

    date: datetime
    rank_limit: int
    uuid: str
    should_compare_with_last_day: bool

    threads: List[ThreadStats] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'threads',
                           self.db.get_daily_threads(self.date))
        object.__setattr__(self, 'counts', Counts(self.threads))

    def generate(self) -> Tuple[str, str, str]:
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

        title = self.date.strftime("日度趋势 %Y-%m-%d")
        name = "页 ❬1 / 1❭"

        content = self._generate_head() + '\n'
        daily_qst_reference = self._generate_daily_qst_reference()
        if daily_qst_reference is not None:
            content += daily_qst_reference + '\n'
        content += self._generate_summary() + '\n'

        content += '\n'.join([self._format_heading("趋势"), '', ''])
        content += self._generate_trending_board() + '\n'

        misc_content = self._generate_misc()
        if misc_content is not None:
            content += '\n'.join([self._format_heading("杂项"), '', ''])
            content += misc_content + '\n'

        content += '\n'.join([self._format_heading("META"), '', ''])
        content += self._generate_meta() + '\n'

        return [title, name, content]

    def _format_heading(self, name) -> str:
        return f"{MAIN_DIVIDER_PART}　{name}　{MAIN_DIVIDER_PART}"

    def _generate_head(self) -> str:
        return '\n'.join([
            self.date.strftime(f"【 {ZWSP} 跑团版 趋势 日度报告〔%Y-%m-%d〕】"),
            f"统计范围：当日上午4时～次日上午4时前",
            "页 ❬1 / 1❭", '',
        ])

    def _generate_daily_qst_reference(self) -> Optional[str]:
        daily_qst = self.db.get_daily_qst(self.date, DAILY_QST_THREAD_ID)
        if daily_qst is None:
            return None
        return '\n'.join([
            f"当期跑团日报：>>No.{daily_qst[0]}（位于原串第{(daily_qst[1]-1)//19+1}页）", '',
        ])

    def _generate_summary(self) -> str:
        class AttrsNone:
            def __getattribute__(self, _):
                return None
        counts_before = AttrsNone()
        if self.should_compare_with_last_day:
            one_day_before = self.date - timedelta(days=1)
            counts_before = Counts(self.db.get_daily_threads(one_day_before))

        count_texts = self.__format_counts(self.counts, counts_before)

        return '\n'.join(["统计范围内："] + list(map(lambda x: f"{ZWSP} ∗ {x}", count_texts))) + '\n'

    def __format_counts(self, counts: Counts, counts_before: Counts) -> List[str]:
        return [
            f"总计出现主题串 {self.__format_value_with_delta(counts.threads, counts_before.threads)} 串",
            f"新增主题串 {self.__format_value_with_delta(counts.new_threads, counts_before.new_threads)} 串",
            f"新增回应 {self.__format_value_with_delta(counts.new_posts, counts_before.new_posts)} 条",
            f"主题串新增回应 {self.__format_q(counts.thread_new_post_quartiles, counts_before.thread_new_post_quartiles)}"
            # 没太大意义…
            # f"平均主题串新增回应 {counts.thread_new_post_average} 条，"
            # + f"中位 {counts.thread_new_post_median} 条，"
            # + f"S²={counts.thread_new_post_variance}"
        ]

    def __format_value_with_delta(self, value: int, old_value: Optional[int]) -> str:
        if old_value is None:
            return str(value)
        delta = value - old_value
        if delta > 0:
            return f"{value}(↑{delta})"
        elif delta < 0:
            return f"{value}(↓{abs(delta)})"
        return f"{value}(→0)"

    def __format_q(self, q: List[float], old_q: List[float]) -> str:
        if old_q is None:
            old_q = [None] * len(q)
        q_texts = [f"Q₁={self.__format_value_with_delta(q[0], old_q[0])}"]
        q_texts += [f"中位数={self.__format_value_with_delta(q[1], old_q[1])}"]
        q_texts += [f"Q₃={self.__format_value_with_delta(q[2], old_q[2])}"]
        return ' '.join(q_texts)

    def _generate_trending_board(self) -> str:
        lines = []
        for (i, thread) in enumerate(self.threads):
            rank = i + 1
            if rank > self.rank_limit \
                    and thread.increased_response_count != self.threads[i-1].increased_response_count:
                break
            lines += [self.__generate_thread_entry(thread, rank)]

        return '\n'.join(lines)

    def __generate_thread_entry(self, thread: ThreadStats, rank: int) -> str:
        head = f"#{rank:02d}"
        if thread.is_new:
            head += f" [+{thread.increased_response_count} 回应 NEW!]"
        else:
            head += f" [+{thread.increased_response_count} ={thread.total_reply_count} 回应]"
        head += f" [@{thread.created_at.strftime('%Y-%m-%d')}" \
            + f"{ thread.created_at.strftime(' %H:%M') if thread.is_new else ''}]"

        subhead = []
        subhead += [f"(PO回应={thread.increased_response_count_by_po})"]
        approx_distinct_cookie_count = thread.distinct_cookie_count//5*5
        if approx_distinct_cookie_count != 0:
            subhead += [f"(参与饼干≥{approx_distinct_cookie_count})"]
        else:
            subhead += [f"(参与饼干>0)"]

        return '\n'.join([
            head,
            "　"*2 + ' '.join(subhead),
            f">>No.{thread.id}",
            thread.generate_summary(free_lines=3),
            ZWSP.join([f"━━━━"]*4),
            '',
        ])

    def _generate_misc(self) -> Optional[str]:
        entries = list(filter(lambda x: x is not None, [
            self._generate_tail_frequencies_report(),
            self._generate_consecutive_tails_report(),
        ]))
        if len(entries) > 0:
            return '\n'.join(entries)
        return None

    def _generate_tail_frequencies_report(self) -> Optional[str]:
        (count, tail_frequencies) = self.db.get_tail_frequencies(self.date)

        if count == 0:
            return None

        text = f"此间，「r」串尾出目频率 (n={count})：\n"
        if 0 in tail_frequencies:
            tail_frequencies.move_to_end(0)
        f = list(reversed(tail_frequencies.items()))
        f_max_min = [max(f, key=lambda x: x[1])[0],
                     min(f, key=lambda x: x[1])[0]]
        f = list(map(lambda x: (
            "{}={:05.2f}%*" if x[0] in f_max_min else "{}={:05.2f}% ").format(*x), f))
        for i in range(0, 10, 4):
            text += ' '.join(f[i:i+4]) + ("（*最高/最低）" if i == 8 else "") + '\n'
        return text

    def _generate_consecutive_tails_report(self) -> Optional[str]:
        lucky_numbers = self.db.get_consecutive_tail_counts(self.date, 3)
        if len(lucky_numbers) == 0:
            return None
        lines = []
        for (n, count, zero_count) in lucky_numbers:
            text = "{} 连号 {} 次 ({:.2f}‰)，".format(
                n, count, count / self.counts.new_posts * 1000)
            if zero_count > 0:
                text += "其中全 0 有 {} 次 ({:.2f}‰)".format(
                    zero_count, zero_count / self.counts.new_posts * 1000)
            else:
                text += "其中没有全 0"
            lines += [text]
        return '\n'.join(["此间，串尾连号次数：", "；\n".join(lines) + "。", ''])

    def _generate_meta(self) -> str:
        stats = self.db.get_meta_stats(self.date)

        return '\n'.join([
            f"统计期间：共上传 {stats.total_bandwidth_usage[0]:,} 字节，"
            + f"下载 {stats.total_bandwidth_usage[1]:,} 字节。", '',
            f"UUID={self.uuid} # 定位用", ''
        ])


if __name__ == '__main__':
    main()
