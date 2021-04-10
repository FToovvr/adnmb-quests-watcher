#!/usr/bin/env python3

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import traceback

import os
from datetime import datetime, timedelta
import json
import logging
import logging.config
import traceback

import psycopg2

import anobbsclient
from anobbsclient.walk import create_walker, BoardWalkTarget, ReversalThreadWalkTarget

from commons.consts import local_tz
from commons.config import load_config
from models.activity import Activity, TotalBandwidthUsage, Stats
from models.collecting import DB


# 默认单线程

logging.config.fileConfig('logging.1_collect.conf')


def main():

    config = load_config('./config.yaml')

    with psycopg2.connect(config.database.connection_string) as conn_activity, \
            psycopg2.connect(config.database.connection_string) as conn_db:
        activity = Activity(conn=conn_activity,
                            activity_type='collect',
                            run_at=datetime.now(tz=local_tz))
        db = DB(conn=conn_db,
                completion_registry_thread_id=config.completion_registry_thread_id)

        stats = Stats()

        fetching_since = db.should_collect_since

        is_successful = False
        message = None
        try:
            fetch_board(db=db, activity=activity, client=config.client.create_client(),
                        board_id=config.board_id, fetching_since=fetching_since, stats=stats)
            is_successful = True
        except:
            exc_text = traceback.format_exc()
            logging.critical(exc_text)
            message = exc_text
        finally:
            activity.report_end(is_successful, message, stats)

    if is_successful:
        logging.info("成功结束")
    else:
        exit(1)


def fetch_board(db: DB, activity: Activity, client: anobbsclient.Client,
                board_id: int, fetching_since: datetime, stats: Stats):

    logger = logging.getLogger('FETCH')

    walker = create_walker(
        target=BoardWalkTarget(
            board_id=board_id,
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

    now = datetime.now(tz=local_tz)

    for (i, thread) in enumerate(threads_on_board):
        logger.debug(f'串 #{i}。串号 = {thread.id}，'
                     + f'最后修改时间 = {thread.last_modified_time}')

        if is_first_found_thread:
            is_first_found_thread = False
            activity.report_collecting_range(
                since=fetching_since, until=thread.last_modified_time)

        is_thread_recorded = db.is_thread_recorded(thread.id)
        if not is_thread_recorded:
            stats.new_thread_count += 1
        # 记录或更新串
        # current_reply_count 在后面一同记录
        db.record_thread(thread, board_id=board_id, updated_at=now)

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

        new_responses_in_preview = list(
            [post for post in thread.replies if is_target(post)])
        if thread.total_reply_count <= 5 \
                or not is_target(thread.replies[0]):
            # 要抓取的内容全在预览里，不用再进串里去翻了
            # TODO 判断是否没有剩余回应（len(thread.total_reply_count) <= 5）应该在 API 那边进行
            if len(new_responses_in_preview) > 0:
                if is_thread_recorded:
                    stats.affected_thread_count += 1
                stats.new_post_count += len(new_responses_in_preview)
            db.record_thread_replies(thread=thread, replies=new_responses_in_preview,
                                     total_reply_count=thread.total_reply_count,
                                     updated_at=now)
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
            targets += new_responses_in_preview
            now_after_fetching_inside_thread = datetime.now(tz=local_tz)
            db.record_thread_replies(thread=thread, replies=targets,
                                     total_reply_count=final_reply_count,
                                     updated_at=now_after_fetching_inside_thread)
            stats.total_bandwidth_usage.add(bandwidth_usage_for_thread.total)
            if len(targets) > 0:
                if is_thread_recorded:
                    stats.affected_thread_count += 1
                stats.new_post_count += len(targets)
            logger.debug(f'串 #{i} 已抓取到范围内所有新回应，记录后到此结束。'
                         + f'遍历访问页数 = {thread_walk_page_count}，'
                         + f'期间 (上传字节数, 下载字节数) = {bandwidth_usage_for_thread.total}')


if __name__ == '__main__':
    main()
