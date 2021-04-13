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

    uuid: Optional[str] = None

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
            cur.execute(r'''SELECT * FROM get_publication_record_id_and_create_record_if_needed(%s, %s, %s)''',
                        (self.subject_date, self.report_type, self.uuid))
            object.__setattr__(self, '_id', cur.fetchone()[0])
            cur.execute(r'''SELECT * FROM get_publication_record_uuid(%s)''',
                        (self._id,))
            uuid = cur.fetchone()[0]
            if (not self.uuid):
                object.__setattr__(self, 'uuid', uuid)
            else:
                assert(self.uuid == uuid)

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
                        (self._id,))
            return cur.fetchone()[0]

    def increase_attempts(self):
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''CALL increase_publication_record_attempts(%s)''',
                        (self._id,))

    @property
    def reply_posts(self) -> List[PublishedPost]:
        posts = []
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
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
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''CALL report_publication_destination_thread_id_and_page_count(%s, %s, %s)''',
                        (self._id, thread_id, reply_count))

    def report_found_reply_post(self, report_page_number: int, post_id: int, offset: int):
        with self.conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''CALL report_found_publication_page(%s, %s, %s, %s)''',
                        (self._id, report_page_number, post_id, offset))
