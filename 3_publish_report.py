#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Tuple, List, Dict, OrderedDict, Optional, Union
from dataclasses import dataclass, field

import argparse
from pathlib import Path
import sys
import os
import sqlite3
from datetime import datetime, timedelta, time, date
import logging
import logging.config
import traceback
import re
from time import sleep

import requests
from bs4 import BeautifulSoup

import anobbsclient
from anobbsclient.walk import create_walker, ReversalThreadWalkTarget

from commons import get_client, Trace, local_tz, ZWSP, OMITTING, get_target_date
from commons.stat_model import ThreadStats, Counts, Stats, DB
from commons.debugging import super_huge_thread


FORMAT_VERSION = '2.0'
CHARACTER_COUNT_METHOD_VERSION = '2'
CHARACTER_COUNT_METHOD_EXPLANATION = "除换行与一般空白外字符的个数"

DAILY_QST_THREAD_ID = int(os.environ.get(
    'ANOBBS_QUESTS_DAILY_QST_THREAD_ID', -1))
DAILY_QST_THREAD_ID = None if DAILY_QST_THREAD_ID is None else DAILY_QST_THREAD_ID


@dataclass(frozen=True)
class Arugments:
    target_date: date
    publish_on_thread: Optional[int]
    check_sage: bool
    notify_thread: Optional[int]
    page_capacity: int
    including: including
    no_compare: bool

    db_path: Path

    force_slience: bool
    force_no_publish: bool

    @property
    def needs_to_track(self) -> bool:
        return self.publish_on_thread is not None \
            and not self.force_no_publish

    @property
    def requires_client(self) -> bool:
        return self.publish_on_thread is not None \
            or self.check_sage


@dataclass(frozen=True)
class Including:
    method: str
    arguments: List[Any]

    def __str__(self) -> str:
        if self.method == 'top':
            return f"前 {self.arguments[0]} 位"
        elif self.method == 'not_below_q2':
            return "前 50%"
        elif self.method == 'not_below_q3':
            return "前 25%"
        elif self.method == 'count_at_least':
            return f"新增回应≥{self.arguments[0]}"
        else:
            assert(False)


def parse_args(args: List[str]) -> Arugments:
    parser = argparse.ArgumentParser(
        description="发布报告。"
    )

    parser.add_argument(
        type=date.fromisoformat, nargs='?',
        dest='target_date',
        help="要报告的日期",
    )
    parser.add_argument(
        '--publish-on-thread', type=int, default=None,
        dest='publish_on_thread',
        help="要发布到的串",
    )
    parser.add_argument(
        '--check-sage', action='store_true',
        dest='check_sage',
        help="是否在发布前检查串有无被下沉",
    )
    parser.add_argument(
        '--notify-thread', type=int, default=None,
        dest='notify_thread',
        help="成功发布后要通知的串",
    )
    parser.add_argument(
        '--page-capacity', type=int, default=20,
        dest='page_capacity',
        help="每页最大能容纳主题串的数量",
    )
    parser.add_argument(
        '--including', type=str, nargs='+', default=['not_below_q3'],
        dest='including',
        help="主题串包含进报告的条件",
    )
    parser.add_argument(
        '--no-compare', action='store_true',
        dest='no_compare',
        help="报告不要包含与之前的数据的比较",
    )
    parser.add_argument(
        '--db-path', type=str, default='db.sqlite3',
        dest='db_path',
        help="DB 所在的路径",
    )
    parser.add_argument(
        '--debugging-scenario', '-D', type=str, default=None,
        choices=['none', 'preview', 'preview-',
                 'publish_only', 'notify', 'notify-', 'check_sage'],
        dest='debugging_scenario',
        help="除错用。根据所输入的值，可能会修改其他参数的内容",
    )

    parsed = parser.parse_args(args)

    if parsed.target_date is None:
        parsed.target_date = get_target_date()

    if parsed.check_sage or parsed.notify_thread is not None:
        assert(parsed.publish_on_thread is not None)

    if parsed.including[0] in ['not_below_q2', 'not_below_q3']:
        assert(len(parsed.including) == 1)
        parsed.including = Including(parsed.including[0], [])
    elif parsed.including[0] == 'top':
        assert(len(parsed.including) <= 2)
        if len(parsed.including) == 1:
            parsed.including = Including('top', [32])
        else:
            parsed.including = Including('top', [int(parsed.including[1])])
    elif parsed.including[0] == 'count_at_least':
        assert(len(parsed.including) <= 2)
        if len(parsed.including) == 1:
            parsed.including = Including('count_at_least', [20])
        else:
            parsed.including = Including(
                'count_at_least', [int(parsed.including[1])])
    else:
        assert(False)

    parsed.db_path = Path(parsed.db_path)
    if not parsed.db_path.exists():
        print(f"{args.db_path} 不存在，将终止")
        exit(1)

    force_slience = False
    force_no_publish = False
    s = parsed.debugging_scenario or 'none'
    if s.startswith('preview'):
        # 只预览产生的报告
        parsed.publish_on_thread = None
        parsed.check_sage = False
    elif s.startswith('check_sage'):
        # 只执行检查是否下沉
        assert(parsed.publish_on_thread is not None)
        parsed.check_sage = True
        force_slience = True
        force_no_publish = True
    elif s.startswith('publish_only'):
        # 只发布报告
        assert(parsed.publish_on_thread is not None)
        parsed.notify_thread = None
        parsed.check_sage = False
    elif s.startswith('notify'):
        # 发布通知，但是会通知到与报告相同的串内
        assert(parsed.publish_on_thread is not None)
        parsed.notify_thread = parsed.publish_on_thread
        parsed.check_sage = False
    else:
        assert(s == 'none')
    if s.endswith('-'):
        parsed.including = Including('top', [1])

    args_obj = Arugments(
        target_date=parsed.target_date,
        publish_on_thread=parsed.publish_on_thread,
        check_sage=parsed.check_sage,
        notify_thread=parsed.notify_thread,
        page_capacity=parsed.page_capacity,
        including=parsed.including,
        no_compare=parsed.no_compare,

        db_path=parsed.db_path,

        force_slience=force_slience,
        force_no_publish=force_no_publish,
    )

    return args_obj


# 收录条件
def RANK_INCLUDING(thread: ThreadStats, including: Including,
                   threads: List[ThreadStats], counts: Counts,):
    if including.method == 'top':
        nth_thread = threads[including.arguments[0] - 1] \
            if len(threads) >= including.arguments[0] else threads[-1]
        return thread.increased_response_count >= nth_thread.increased_response_count
    elif including.method == 'not_below_q2':
        return thread.increased_response_count >= counts.thread_new_post_quartiles[1]
    elif including.method == 'not_below_q3':
        return thread.increased_response_count >= counts.thread_new_post_quartiles[2]
    elif including.method == 'count_at_least':
        return thread.increased_response_count >= including.arguments[0]
    else:
        assert(False)


# 每页收录数目
RANK_PAGE_CAPACITY = 20

MAIN_DIVIDER_PART = f"══{ZWSP}══{ZWSP}══"
META_MAIN_DIVIDER = f"{MAIN_DIVIDER_PART}　META　{MAIN_DIVIDER_PART}"


def main():

    args = parse_args(sys.argv[1:])

    if args.requires_client:
        client = get_client()
    else:
        client = None

    if args.needs_to_track:
        trace = Trace(conn=sqlite3.connect(args.db_path),
                      date=args.target_date, type_='trend')
        attempts = trace.attempts
        if trace.is_done or attempts > 3:
            return
        trace.increase_attempts()

        logging.config.fileConfig('logging.3_publish_report.conf')

        logging.info(f"开始进行发布报告相关流程。UUID={trace.uuid}")

    if args.check_sage:
        (trend_thread, _) = client.get_thread_page(
            args.publish_on_thread, page=1)
        if trend_thread.marked_sage:
            logging.warn("趋势串已下沉。本次终止")

    if args.needs_to_track:
        logging.info("尚未发送回应请求以发布报告，将生成报告文本并尝试发送回应请求")
        uuid = trace.uuid
    else:
        uuid = None

    pages = retrieve_data_then_generate_trend_report_text(
        date=args.target_date, uuid=uuid,
        rank_inclusion_method=args.including,
        should_compare_with_last_day=not args.no_compare,
    )

    if not args.needs_to_track:
        if args.force_slience:
            return
        for (title, name, content) in pages:
            print('\n'.join([
                "标题：" + title,
                "名称：" + name,
                content,
            ]))
        return

    logging.info(f"报告文本页数：{len(pages)}")

    trace.report_thread_id_and_reply_count(
        thread_id=args.publish_on_thread,
        reply_count=len(pages)
    )
    first_rount = True
    for post in trace.reply_posts:
        logging.info(f"处理发布第 {post.report_page_number} 页…")
        if post.reply_post_id is not None:
            logging.info(f"本页已有发布成功的记录，跳过")
            continue

        if first_rount:
            first_rount = False
        else:
            logging.info(f"在发送报告前，由于发串间隔限制，将等待30秒")
            sleep(30)

        (title, name, content) = pages[post.report_page_number-1]

        try:
            client.reply_thread(content, to_thread_id=args.publish_on_thread,
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

        logging.info(f"将查找属于本页报告的回应")

        found_post = find_last_post_with_uuid(client, args.publish_on_thread)
        if found_post is None:
            logging.error("未找到任何带有 UUID 回应，本次终止")
            exit(1)

        (report_page_number, post_id, uuid, offset) = found_post
        if uuid != trace.uuid:
            logging.error(f"最后带有 UUID 的回应的 UUID 与本次的不匹配，本次终止。找到的 UUID={uuid}")
            exit(1)
        if report_page_number != post.report_page_number:
            logging.error("最后带有 UUID 的回应的页数与本次的不匹配，本次终止。" +
                          f"找到的页数={report_page_number}")
            exit(1)

        logging.info(f"找到本页报告对应回应，将记录。回应串号={post_id}，偏移={offset}")

        trace.report_found_reply_post(
            report_page_number=report_page_number,
            post_id=post_id, offset=offset,
        )

    logging.info("已发送各页报告且找到报告各页对应的各回应")

    if args.notify_thread is not None:
        # TODO: 检查成功与否
        notify_to_thread_id = args.notify_thread
        logging.info(f"将发送报告出炉通知。由于发串间隔限制，将等待30秒")
        sleep(30)

        posts = trace.reply_posts
        content = args.target_date.strftime(
            f"%Y年%-m月%-d日 跑团版 趋势日度报告：\n")
        content += '\n'.join(
            list(map(lambda x: f">>No.{x.reply_post_id}", posts))
        ) + '\n'
        min_reply_pn = (posts[0].reply_offset-1)//19+1
        max_reply_pn = (posts[-1].reply_offset-1)//19+1
        if min_reply_pn == max_reply_pn:
            content += f"(位于原串第{min_reply_pn}页)"
        else:
            content += f"(位于原串第{min_reply_pn}〜{max_reply_pn}页)"
        content += '\n'

        client.reply_thread(
            to_thread_id=notify_to_thread_id,
            title="本期跑团版趋势报告出炉",
            name=args.target_date.strftime("%Y年%-m月%-d日 号"),
            content=content,
        )

    logging.info("成功结束")


def find_last_post_with_uuid(client: anobbsclient.Client, thread_id: int) -> Optional[Tuple[int, int, str, int]]:
    """
    Returns
    -------
    [0] : int
        报告的页数。

        不是回应所在的页数
    [1] : int
        目标回应的串号。
    [2] : str
        找到的 UUID。
    [3] : int
        目标回应的偏移。

    如果没找到或者找到的第一个 uuid 不匹配，返回 None。
    """

    # TODO: 可以根据上一次回应所在位置预测一下，大部分情况能把请求减少到1次
    # TODO: 如果发现串 SAGE 了，以后就不发了，或者提前检查一下有无 SAGE？

    (page_1, _) = client.get_thread_page(id=thread_id, page=1, for_analysis=1)
    page_1: anobbsclient.ThreadPage = page_1
    # TODO: 其实这个可以在 API 那边定义 property 来算吧
    total_pages = (page_1.body.total_reply_count - 1) // 19 + 1

    walker = create_walker(
        target=ReversalThreadWalkTarget(
            thread_id=thread_id,
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
                r"(?:.*\n)+Report ID = ([0-9a-f\-]+).*(?:\n.*)*",
                re.MULTILINE,
            )
            result = uuid_rx.match(text)
            if result is None:
                continue
            uuid = result.group(1)

            report_pn = int(re.match(r"页 ❬(\d+) / \d+❭", post.name).group(1))

            return (report_pn, post.id, uuid, (pn-1)*19+1+(len(page.replies)-1-i))

    return None


def retrieve_data_then_generate_trend_report_text(date: datetime, uuid: str,
                                                  rank_inclusion_method: Including,
                                                  should_compare_with_last_day: bool,
                                                  ) -> Tuple[str, str, str]:
    with sqlite3.connect('file:db.sqlite3?mode=ro', uri=True) as conn:
        db = DB(conn=conn)
        return TrendReportTextGenerator(
            db=db,
            date=date,
            rank_inclusion_method=rank_inclusion_method,
            rank_page_capacity=RANK_PAGE_CAPACITY,
            uuid=uuid,
            should_compare_with_last_day=should_compare_with_last_day,
        ).generate()


@dataclass(frozen=True)
class TrendReportTextGenerator:

    db: DB

    date: datetime
    rank_inclusion_method: Including
    rank_page_capacity: int
    uuid: str
    should_compare_with_last_day: bool

    threads: List[ThreadStats] = field(init=False)
    counts: Counts = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'threads',
                           self.db.get_daily_threads(self.date))
        object.__setattr__(self, 'counts', Counts(self.threads))

    def generate(self) -> List[Tuple[str, str, str]]:
        """
        Returns
        -------
        [n][0] : str
            标题。
        [n][1] : str
            名称。
        [n][2] : str
            正文。
        """

        trending_boards \
            = self._generate_trending_boards(self.rank_page_capacity)
        report_pages = []
        title = self.date.strftime("日度趋势 %Y-%m-%d")
        for (i, board) in enumerate(trending_boards):
            page_number = i + 1
            name = f"页 ❬{page_number} / {len(trending_boards)}❭"
            page = self._generate_page(
                board, page_number, len(trending_boards))
            report_pages.append([title, name, page])
        return report_pages

    def _generate_page(self, trending_board: str, page_number: int, total_page_number: int) -> str:

        content = self._generate_head(page_number, total_page_number) + '\n'
        if page_number == 1:
            daily_qst_reference = self._generate_daily_qst_reference()
            if daily_qst_reference is not None:
                content += daily_qst_reference + '\n'
            content += self._generate_summary() + '\n'
            content += "收录范围：" + str(self.rank_inclusion_method)
            content += '\n\n'

            content += '\n'.join([self._format_heading("　说明　"), '', ''])
            content += f"「+X/Y」：\n{ZWSP}  「X」代表总增量，「Y」代表PO增量。\n"
            content += '\n'
            content += f"文本统计方式：\n{ZWSP}  {CHARACTER_COUNT_METHOD_EXPLANATION}。\n"
            content += '\n'

        content += '\n'.join([self._format_heading("　趋势　"), '', ''])
        content += trending_board + '\n'

        if page_number == 1:
            misc_content = self._generate_misc()
            if misc_content is not None:
                content += '\n'.join([self._format_heading("　杂项　"), '', ''])
                content += misc_content + '\n'

        content += '\n'.join([self._format_heading("　META　"), '', ''])
        content += self._generate_meta(page_number) + '\n'

        return content

    def _format_heading(self, name) -> str:
        return f"{MAIN_DIVIDER_PART}{name}{MAIN_DIVIDER_PART}"

    def _generate_head(self, page_number: int, total_page_number: int) -> str:
        return '\n'.join([
            self.date.strftime(f"【 {ZWSP} 跑团版 趋势 日度报告〔%Y-%m-%d〕】"),
            f"　 {ZWSP} 第 ❬{page_number} / {total_page_number}❭ 页",
            f"统计范围：当日上午4时～次日上午4时前",
            '',
        ])

    def _generate_daily_qst_reference(self) -> Optional[str]:
        if DAILY_QST_THREAD_ID is None:
            return None
        daily_qst = self.db.get_daily_qst(self.date, DAILY_QST_THREAD_ID)
        if daily_qst is None:
            return None
        return '\n'.join([
            f"当期跑团日报：>>No.{daily_qst[0]} (位于原串第{(daily_qst[1]-1)//19+1}页)", '',
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

    def _generate_trending_boards(self, step: int) -> List[str]:
        included_threads = []
        threads_with_new_blue_text = []
        for (i, thread) in enumerate(self.threads):
            rank = i+1
            if thread.has_new_blue_text:
                threads_with_new_blue_text.append([rank, thread])
            elif RANK_INCLUDING(thread, self.rank_inclusion_method, self.threads, self.counts):
                included_threads.append([rank, thread])
        threads_with_new_blue_text = sorted(threads_with_new_blue_text,
                                            key=lambda x: x[1].created_at)
        included_threads = threads_with_new_blue_text + included_threads

        boards = []
        for i in range(0, len(included_threads), step):
            board = self._generate_trending_board(
                included_threads[i:i+step], i)
            boards.append(board)
        return boards

    def _generate_trending_board(self, threads: List[Tuple[int, ThreadStats]], i_start: int) -> str:
        lines = []
        for (i, [rank, thread]) in enumerate(threads):
            lines += [self.__generate_thread_entry(thread, rank)]

        return '\n'.join(lines)

    def __generate_thread_entry(self, thread: ThreadStats, rank: int) -> str:
        # thread = super_huge_thread  # DEBUGGING
        head = f"#{rank}"
        padding = len(head) + 1
        if thread.is_new:
            head += f" [+{thread.increased_response_count}/{thread.increased_response_count_by_po} 回应 NEW!]"
        else:
            head += f" [+{thread.increased_response_count}/{thread.increased_response_count_by_po} ={thread.total_reply_count} 回应]"
        head += f" [@"
        if thread.is_new:
            # # TODO: 应该更严谨些
            # if thread.created_at.day == self.date.day:
            #     head += "当日"
            # else:
            #     head += "次日"
            head += thread.created_at.strftime('%m-%d')
        else:
            head += thread.created_at.strftime('%Y-%m-%d')
        head += (thread.created_at.strftime(' %H:%M')
                 if thread.is_new else '') + "]"

        subhead_lines = []

        subhead_1 = []
        approx_distinct_cookie_count = thread.distinct_cookie_count//5*5
        if approx_distinct_cookie_count != 0:
            subhead_1 += [f"(参与饼干≥{approx_distinct_cookie_count})"]
        else:
            subhead_1 += [f"(参与饼干>0)"]
        character_count = "(+" + \
            f"{thread.increased_character_count/1000:.2f}K" + "/"
        if thread.increased_character_count_by_po != 0:
            character_count += f"{thread.increased_character_count_by_po/1000:.2f}K"
        else:
            character_count += "0"
        character_count += " 文本)"
        subhead_1 += [character_count]
        subhead_lines += [' '.join(subhead_1)]

        if not self.db.is_thread_disappeared(thread.id):
            blue_text = thread.blue_text
            if blue_text is not None:
                blue_text = blue_text.strip()
                if len(blue_text) > 8:
                    blue_text = blue_text[:8] + OMITTING
                if thread.has_new_blue_text:
                    subhead_lines += [f"(新蓝字！「{blue_text}」)"]
                else:
                    subhead_lines += [f"(蓝字「{blue_text}」)"]

            preview = thread.generate_summary(free_lines=3)
        else:
            subhead_lines += ["(已消失)"]
            preview = None

        return '\n'.join(
            [head]
            + list(map(lambda x: f'{ZWSP} ' * padding + x, subhead_lines))
            + [f">>No.{thread.id}"]
            + ([preview] if preview is not None else [])
            + [ZWSP.join([f"━━━━"]*4), '']
        )

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

        text = f"统计范围内，「r」串尾出目频率 (n={count})：\n"
        if 0 in tail_frequencies:
            tail_frequencies.move_to_end(0)
        f = list(reversed(tail_frequencies.items()))
        f_max_min = [max(f, key=lambda x: x[1])[0],
                     min(f, key=lambda x: x[1])[0]]
        f = list(map(lambda x: (
            "{}={:05.2f}%*" if x[0] in f_max_min else "{}={:05.2f}% ").format(*x), f))
        lines = []
        for i in range(0, 10, 4):
            lines += [' '.join(f[i:i+4])]
        lines[-1] = lines[-1].rstrip()
        lines[-1] += "（*最高/最低）"
        return text + '\n'.join(lines) + '\n'

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
        return '\n'.join(["统计范围内，串尾连号次数：", "；\n".join(lines) + "。", ''])

    def _generate_meta(self, page_number: int) -> str:
        stats = self.db.get_meta_stats(self.date)

        lines = []
        if page_number == 1:
            lines += [
                f"统计期间：共上传 {stats.total_bandwidth_usage[0]:,} 字节，"
                + f"下载 {stats.total_bandwidth_usage[1]:,} 字节。", '',
            ]
        lines += [
            f'文本统计方式 Version = {CHARACTER_COUNT_METHOD_VERSION} ', '']
        lines += [f'Format Version = {FORMAT_VERSION}', '']
        lines += [f"Report ID = {self.uuid} # 定位用", '']
        return '\n'.join(lines)


if __name__ == '__main__':
    main()
