#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass

import sqlite3
from datetime import datetime, timedelta, time

from dateutil import tz

from bs4 import BeautifulSoup

import anobbsclient

local_tz = tz.gettz('Asia/Shanghai')

MAIN_DIVIDER_PART = "══" + '\u200b' + "══" + '\u200b' + "══"
SUB_DIVIDER = ("━━━━" + '\u200b' + "━━━━") * 2


def main():

    now = datetime.now(tz=local_tz)
    assert(now.time() >= time(hour=4))
    yesterday = now - timedelta(days=1)

    db = DB(conn=sqlite3.connect('file:db.sqlite3?mode=ro', uri=True))
    (threads, stats) = db.get_daily_threads(yesterday)
    db.close()

    trend_report_text = generate_trend_report_text(yesterday, threads, stats)
    print(trend_report_text)


def generate_trend_report_text(date: datetime, threads: List[ThreadStats], stats: Stats):

    new_threads = list(filter(lambda x: x.is_new, threads))
    new_thread_count = len(new_threads)
    new_post_count = sum(
        list(map(lambda x: x.increased_response_count, threads)))

    lines = [
        date.strftime("日度报告 趋势 %Y-%m-%d"),
        date.strftime("统计范围：当日上午4时～次日上午4时前")
    ]

    lines += [f"统计范围内：新增主题串{new_thread_count}个，新增回应{new_post_count}条。"]

    lines += ['', f"{MAIN_DIVIDER_PART}　趋势　{MAIN_DIVIDER_PART}"]

    for (i, thread) in enumerate(threads):
        rank = i + 1
        if rank > 32:
            break

        lines += ['']

        head = f"#{('%02d' % rank)}"
        head += f" [+{thread.increased_response_count}回应 ={thread.total_reply_count}回应]"
        head += f" [@{thread.created_at.strftime('%Y-%m-%d')}{'🆕' if thread.is_new else ''}]"
        lines += [head]

        lines += [f">>No.{thread.id}"]

        lines += thread.generate_summary(free_lines=3).split('\n')

        lines += [SUB_DIVIDER]

    lines += ['', f"{MAIN_DIVIDER_PART}　META　{MAIN_DIVIDER_PART}"]
    lines += [f"统计期间：共上传{stats.total_bandwidth_usage[0]}字节，下载{stats.total_bandwidth_usage[1]}字节。"]

    return "\n".join(lines)


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

    @property
    def content(self) -> str:
        return BeautifulSoup(self.raw_content, features='html.parser').get_text()

    def generate_summary(self, free_lines: int) -> str:
        lines = []
        if self.title is not None:
            free_lines -= 1
            lines += [f"标题：{self.title}"]
        if self.name is not None:
            free_lines -= 1
            lines += [f"名称：{self.name}"]
        for content_line in self.content.split('\n'):
            if free_lines == 0:
                lines += ["…"]
                break
            content_line = content_line.rstrip()
            line_to_add = ""
            for line_part in [content_line[i: i+16] for i in range(0, len(content_line), 18)]:
                if free_lines == 0:
                    line_to_add += "…"
                    break
                line_to_add += line_part.replace('\u200b', '')
                free_lines -= 1
            lines += [line_to_add]
        while True:
            if lines[-1].strip() == "":
                lines.pop()
            else:
                break

        return "\n".join(lines)


@dataclass(frozen=True)
class Stats:
    total_bandwidth_usage: anobbsclient.BandwidthUsage


@dataclass(frozen=True)
class DB:

    conn: sqlite3.Connection

    def close(self):
        self.conn.close()

    def get_daily_threads(self, date: datetime) -> (List[ThreadStats], Stats):
        lower_bound = date.replace(hour=4, minute=0, second=0, microsecond=0)
        upper_bound = lower_bound + timedelta(days=1)

        rows = self.conn.execute('''
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
                thread.current_reply_count - COALESCE(later_changes.increased_response_count, 0)
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
            ))

        row = self.conn.execute('''
        SELECT 
            sum(uploaded_bytes), sum(downloaded_bytes),
            sum(requested_board_page_count), sum(requested_thread_page_count)
        FROM activity
        WHERE fetched_since >= ? and fetched_since < ?
        ''', (lower_bound.timestamp(), upper_bound.timestamp())).fetchone()

        return threads, Stats((row[0], row[1]))


if __name__ == '__main__':
    main()
