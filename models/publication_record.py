from typing import Optional, List
from dataclasses import dataclass, field

from datetime import datetime, date
import uuid
import json

import psycopg2


@dataclass(frozen=True)
class PublishedPost:

    report_page_number: int
    reply_post_id: Optional[int]
    reply_offset: Optional[int]


@dataclass(frozen=True)
class PublicationRecord:

    conn: psycopg2._psycopg.connection

    subject_date: date
    report_type: str

    _id: int = field(init=False)

    @staticmethod
    def is_report_published(conn: psycopg2._psycopg.connection, subject_date: date, report_type: str) -> bool:

        with conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''SELECT * FROM is_report_published(%s, %s)''',
                        (subject_date, report_type))
            return cur.fetchone()[0]

    def __post_init__(self):

        assert(self.report_type in ['trend', 'new_threads'])

        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''SELECT * FROM get_publication_record_id_and_create_record_if_needed(%s, %s)''',
                        (self.subject_date, self.report_type))
            object.__setattr__(self, '_id', cur.fetchone()[0])

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
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''SELECT * FROM get_publication_record_attempts(%s)''',
                        (self._id))
            return cur.fetchone()[0]

    def increase_attempts(self):
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''CALL increase_publication_record_attempts(%s)''',
                        (self._id))

    @property
    def reply_posts(self) -> List[PublishedPost]:
        posts = []
        with self.conn.cursor() as cur:
            cur.execute(r'''SELECT * FROM get_publication_pages_response_info(%s)''',
                        (self._id,))
            for row in cur:
                posts.append(PublishedPost(
                    report_page_number=row[0],
                    reply_post_id=row[1],
                    reply_offset=row[2],
                ))
                return posts

    def report_thread_id_and_reply_count(self, thread_id: int, reply_count: int):
        self.conn_s3.execute(r'''
            UPDATE publishing_trace
            SET to_thread_id = ?
            WHERE id = ?
        ''', (thread_id, self._id))

        # 以防万一
        self.conn_s3.execute(r'''
            DELETE FROM published_post
            WHERE trace_id = ?
        ''', (self._id,))

        for i in range(reply_count):
            page_number = i+1
            self.conn_s3.execute(r'''
                INSERT INTO published_post (trace_id, page_number)
                VALUES (?, ?)
            ''', (self._id, page_number))
        self.conn_s3.commit()

    def report_found_reply_post(self, report_page_number: int, post_id: int, offset: int):
        self.conn_s3.execute(r'''
            UPDATE published_post
            SET
                reply_post_id = ?,
                reply_offset = ?
            WHERE trace_id = ? AND page_number = ?
        ''', (post_id, offset, self._id, report_page_number))
        self.conn_s3.commit()

    @property
    def uuid(self) -> str:
        return self.conn_s3.execute(r'''
            SELECT uuid FROM publishing_trace
            WHERE id = ?
        ''', (self._id,)).fetchone()[0]
