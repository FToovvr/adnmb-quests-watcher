from typing import Optional, List
from dataclasses import dataclass, field

import logging
from datetime import datetime

import psycopg2

import anobbsclient


@dataclass
class TotalBandwidthUsage:
    usages: List[anobbsclient.BandwidthUsage] = field(default_factory=list)

    def add(self, new_usage: anobbsclient.BandwidthUsage):
        self.usages.append(new_usage)

    @property
    def total(self) -> anobbsclient.BandwidthUsage:
        total_uploaded, total_downloaded = 0, 0
        for [uploaded, downloaded] in self.usages:
            total_uploaded += uploaded or 0
            total_downloaded += downloaded or 0
        return [total_uploaded, total_downloaded]


@dataclass
class Stats:

    new_thread_count = 0
    affected_thread_count = 0
    new_post_count = 0

    board_request_count = 0
    thread_request_count = 0
    logged_in_thread_request_count = 0

    total_bandwidth_usage: TotalBandwidthUsage = field(
        default_factory=TotalBandwidthUsage)


@dataclass(frozen=True)
class Activity:

    conn: psycopg2._psycopg.connection

    activity_type: str

    run_at: Optional[datetime] = None

    logger: Optional[logging.Logger] = field(
        default_factory=lambda: logging.getLogger('DB'))  # TODO: 改成 'Activity'?

    activity_id: int = field(init=False)

    def __post_init__(self):
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT * FROM create_new_activity_and_return_id(%s, %s)',
                        (self.activity_type, self.run_at))
            object.__setattr__(self, 'activity_id', cur.fetchone()[0])

        if self.logger:
            self.logger.info(f'已开始新活动。活动 id = {self.activity_id}')

    @staticmethod
    def never_collected(conn: psycopg2._psycopg.connection) -> bool:
        with conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT * FROM never_collected()')
            return cur.fetchone()[0]

    @property
    def should_collect_since(self) -> datetime:
        # XXX: 原来调用这里会同时更新 `fetched_since`，
        # 现在 `fetched_since` 会在 `report_end` 时再更新
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'SELECT * FROM should_collect_since()')
            return cur.fetchone()[0]

    def report_collecting_range(self, since: datetime, until: datetime):
        if self.logger:
            self.logger.info(
                f'正在汇报本次活动抓取时间范围。活动 ID = {self.activity_id}，'
                + f'此下限 = {since}，此上限 = {until}')
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'CALL report_collecting_range(%s, %s, %s)',
                        (self.activity_id, since, until))
        if self.logger:
            self.logger.info(f'已汇报本次活动抓取时间范围。活动 ID = {self.activity_id}')

    def report_end(self, is_successful: bool, message: Optional[str], stats: Stats):
        total_usage = stats.total_bandwidth_usage.total

        if self.logger:
            self.logger.info(
                f'正在汇报本次活动结果。活动 ID = {self.activity_id}，成功 = {is_successful}，'
                + f'上传字节数 = {total_usage[0]}，下载字节数 = {total_usage[1]}，'
                + f'新记录串数 = {stats.new_thread_count}，有新增回应串数 = {stats.affected_thread_count}，'
                + f'新记录回应数 = {stats.new_post_count}，'
                + f'请求版块页面次数 = {stats.board_request_count}，请求串页面次数 = {stats.thread_request_count}，'
                + f'以登录状态请求串页面次数 = {stats.logged_in_thread_request_count}')
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'CALL report_end(' + ', '.join(['%s']*11) + ')',
                        (
                            self.activity_id,
                            is_successful, message,
                            total_usage[0], total_usage[1],
                            stats.new_thread_count, stats.affected_thread_count,
                            stats.new_post_count,
                            stats.board_request_count, stats.thread_request_count,
                            stats.logged_in_thread_request_count,
            ))

        if self.logger:
            self.logger.info(
                f'已汇报本次活动结果。活动 ID = {self.activity_id}')
