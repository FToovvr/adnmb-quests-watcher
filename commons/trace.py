from typing import Optional
from dataclasses import dataclass

import sqlite3
from datetime import datetime, date
import uuid
import json


@dataclass
class Trace:

    conn: sqlite3.Connection

    date: date

    @property
    def iso_date(self) -> str:
        return self.date.strftime('%Y-%m-%d')

    def __post_init__(self):

        row = self.conn.execute('''
            SELECT date FROM publishing_trace
            ORDER BY id LIMIT 1
        ''').fetchone()

        if row is None or date.fromisoformat(row[0]) < self.date:
            # 本日尚未尝试发布报告

            self.conn.execute('''
                INSERT INTO publishing_trace (`date`, uuid)
                VALUES (?, ?)
            ''', (self.date, str(uuid.uuid4())))
            self.conn.commit()

    @property
    def is_done(self) -> bool:
        return self.reply_post_id is not None

    @property
    def attempts(self) -> int:
        return self.conn.execute('''
            SELECT attempts FROM publishing_trace
            WHERE `date` = ?
        ''', (self.date,)).fetchone()[0]

    @attempts.setter
    def attempts(self, value: int):
        self.conn.execute('''
            UPDATE publishing_trace
            SET attempts = ?
            WHERE `date` = ?
        ''', (value, self.date))
        self.conn.commit()

    @property
    def reply_post_id(self) -> Optional[int]:
        return self.conn.execute('''
            SELECT reply_post_id FROM publishing_trace
            WHERE `date` = ?
        ''', (self.date,)).fetchone()[0]

    @property
    def has_made_reply_request(self) -> bool:
        return self.conn.execute('''
            SELECT has_made_reply_request FROM publishing_trace
            WHERE `date` = ?
        ''', (self.date,)).fetchone()[0] == 1

    def report_made_reply_request(self):
        self.conn.execute('''
            UPDATE publishing_trace
            SET has_made_reply_request = TRUE
            WHERE `date` = ?
        ''', (self.date,))
        self.conn.commit()

    def report_found_reply_post(self, thread_id: int, post_id: int, offset: int):
        self.conn.execute('''
            UPDATE publishing_trace
            SET
                has_made_reply_request = TRUE,
                to_thread_id = ?,
                reply_post_id = ?,
                reply_offset = ?
            WHERE `date` = ?
        ''', (thread_id, post_id, offset, self.date,))
        self.conn.commit()

    @property
    def uuid(self) -> str:
        return self.conn.execute('''
            SELECT uuid FROM publishing_trace
            WHERE `date` = ?
        ''', (self.date,)).fetchone()[0]
