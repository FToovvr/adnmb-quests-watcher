from typing import Optional, List
from dataclasses import dataclass, field

import sqlite3
from datetime import datetime, date
import uuid
import json


@dataclass(frozen=True)
class PublishedPost:

    report_page_number: int
    reply_post_id: Optional[int]
    reply_offset: Optional[int]


@dataclass(frozen=True)
class Trace:

    conn: sqlite3.Connection

    date: date
    type_: str

    _id: int = field(init=False)

    @staticmethod
    def has_trace(conn: sqlite3.Connection, date: date) -> bool:
        return conn.execute(r'''
            SELECT count(id) FROM publishing_trace
            WHERE date = ?
        ''', (date,)).fetchone()[0] != 0

    @property
    def iso_date(self) -> str:
        return self.date.strftime('%Y-%m-%d')

    def __post_init__(self):

        assert(self.type_ in ['trend', 'new_threads'])

        row = self.conn.execute(r'''
            SELECT id FROM publishing_trace
            WHERE date = ? AND type = ?
        ''', (self.date, self.type_)).fetchone()

        if row is None:  # 本日尚未尝试发布报告
            self.conn.execute(r'''
                INSERT INTO publishing_trace (`date`, type, uuid)
                VALUES (?, ?, ?)
            ''', (self.date, self.type_, str(uuid.uuid4())))
            self.conn.commit()

        row = self.conn.execute(r'''
            SELECT id FROM publishing_trace
            WHERE date = ? AND type = ?
        ''', (self.date, self.type_)).fetchone()

        object.__setattr__(self, '_id', row[0])

    @property
    def is_done(self) -> bool:
        posts = self.reply_posts
        if len(posts) == 0:
            return False
        for post in posts:
            if post.reply_post_id is None:
                return False
        return True

    @property
    def attempts(self) -> int:
        return self.conn.execute(r'''
            SELECT attempts FROM publishing_trace
            WHERE id = ?
        ''', (self._id,)).fetchone()[0]

    def increase_attempts(self):
        self.conn.execute(r'''
            UPDATE publishing_trace
            SET attempts = ?
            WHERE id = ?
        ''', (self.attempts + 1, self._id))
        self.conn.commit()

    @property
    def reply_posts(self) -> List[PublishedPost]:
        posts = []
        for row in self.conn.execute(r'''
            SELECT
                page_number,
                reply_post_id, reply_offset
            FROM published_post
            WHERE trace_id = ?
        ''', (self._id,)):
            posts.append(PublishedPost(
                report_page_number=row[0],
                reply_post_id=row[1],
                reply_offset=row[2],
            ))
        return posts

    def report_thread_id_and_reply_count(self, thread_id: int, reply_count: int):
        self.conn.execute(r'''
            UPDATE publishing_trace
            SET to_thread_id = ?
            WHERE id = ?
        ''', (thread_id, self._id))

        # 以防万一
        self.conn.execute(r'''
            DELETE FROM published_post
            WHERE trace_id = ?
        ''', (self._id,))

        for i in range(reply_count):
            page_number = i+1
            self.conn.execute(r'''
                INSERT INTO published_post (trace_id, page_number)
                VALUES (?, ?)
            ''', (self._id, page_number))
        self.conn.commit()

    def report_found_reply_post(self, report_page_number: int, post_id: int, offset: int):
        self.conn.execute(r'''
            UPDATE published_post
            SET
                reply_post_id = ?,
                reply_offset = ?
            WHERE trace_id = ? AND page_number = ?
        ''', (post_id, offset, self._id, report_page_number))
        self.conn.commit()

    @property
    def uuid(self) -> str:
        return self.conn.execute(r'''
            SELECT uuid FROM publishing_trace
            WHERE id = ?
        ''', (self._id,)).fetchone()[0]
