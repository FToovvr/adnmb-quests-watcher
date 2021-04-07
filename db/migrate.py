#!/usr/bin/env python3

from typing import Optional

import sys
import json

import sqlite3
import psycopg2

from datetime import datetime, date
from dateutil import tz

import sys
sys.path.append("..")  # noqa

from models.publication_record import PublicationRecord

local_tz = tz.gettz('Asia/Shanghai')


QST_BOARD_ID = 111


def ts2dt(ts):
    return datetime.fromtimestamp(ts, tz=local_tz) if ts else None


def count_rows(conn_s3: sqlite3.Connection, table: str):
    # XXX: 反正也不会有用户输入
    return conn_s3.execute(f'SELECT count(*) FROM {table}').fetchone()[0]


if len(sys.argv) != 3:
    exit(1)


def main():
    conn_s3 = sqlite3.connect(sys.argv[1])
    with psycopg2.connect(sys.argv[2]) as conn_pg:
        conn_pg: psycopg2._psycopg.connection = conn_pg
        with conn_pg.cursor() as cur_pg:
            cur_pg: psycopg2._psycopg.cursor = cur_pg

            cur_pg.execute(
                r'''SELECT set_config('fto.MIGRATING', %s::text, FALSE)''', (True,))
            cur_pg.execute(
                r'''SELECT set_config('fto.COMPLETION_REGISTRY_THREAD_ID', %s::text, FALSE)''', (22762342,))

        for migrate_fn in [
            migrate_activity_table,
            migrate_publication_tables,
            migrate_thread_table,
            migrate_post_table,
        ]:
            print(f"start: {migrate_fn.__name__}")
            migrate_fn(conn_s3, conn_pg)
            print(f"done: {migrate_fn.__name__}")

            print(conn_pg.notifies)

    conn_pg.close()
    conn_s3.close()


def migrate_activity_table(conn_s3: sqlite3.Connection, conn_pg: psycopg2._psycopg.connection):
    with conn_pg.cursor() as cur_pg:
        for [
            id, run_at, fetched_since, ensured_fetched_until, is_successful, message,
            uploaded_bytes, downloaded_bytes, newly_recorded_thread_count, affected_thread_count,
            newly_recorded_post_count, requested_board_page_count, requested_thread_page_count, logged_in_thread_request_count,
        ] in conn_s3.execute(r'SELECT * FROM activity'):
            cur_pg.execute(r'''
            INSERT INTO activity
            VALUES (''' + ', '.join([r'%s']*15) + r''')
            ON CONFLICT DO NOTHING
            ''', (
                id, None, ts2dt(run_at), ts2dt(fetched_since), ts2dt(
                    ensured_fetched_until), bool(is_successful), message,
                uploaded_bytes, downloaded_bytes, newly_recorded_thread_count, affected_thread_count,
                newly_recorded_post_count, requested_board_page_count, requested_thread_page_count, logged_in_thread_request_count,
            ))


def remove_fid_field(misc_fields: Optional[str]) -> Optional[str]:
    if misc_fields is None:
        return None
    misc_fields = json.loads(misc_fields)
    misc_fields.pop('fid', None)
    if len(misc_fields) == 0:
        return None
    else:
        return json.dumps(misc_fields)


def find_updated_at(conn_s3: sqlite3.Connection, id: int, updated_at: Optional[datetime]) -> datetime:
    if updated_at:
        return updated_at

    row = conn_s3.execute(r'''
        SELECT ensured_fetched_until FROM activity
        WHERE ensured_fetched_until > COALESCE(
            (SELECT created_at FROM post WHERE parent_thread_id = ? ORDER BY id ASC),
            (SELECT created_at FROM thread WHERE id = ?)
        )
        ORDER BY ensured_fetched_until ASC
        LIMIT 1
    ''', (id, id)).fetchone()
    return ts2dt(row[0])


def migrate_thread_table(conn_s3: sqlite3.Connection, conn_pg: psycopg2._psycopg.connection):

    with conn_pg.cursor() as cur_pg:
        last = [None, None]
        for [
            id, content,
            name, email, title,
            created_at, user_id, attachment_base, attachment_extension,
            misc_fields,
            not_anymore_at_least_after,
        ] in conn_s3.execute(r'''
            SELECT
                thread_old_revision.id, thread_old_revision.content,
                thread_old_revision.name, thread_old_revision.email, thread_old_revision.title,
                thread.created_at, thread.user_id, thread.attachment_base, thread.attachment_extension,
                thread.misc_fields,
                not_anymore_at_least_after
            FROM thread_old_revision
            LEFT JOIN thread ON thread_old_revision.id = thread.id
            ORDER BY thread_old_revision.id ASC, not_anymore_at_least_after ASC
        '''):
            updated_at = None
            if last[0] == id:
                updated_at = ts2dt(last[1])
            else:
                updated_at = find_updated_at(conn_s3, id, None)
            misc_fields = remove_fid_field(misc_fields)
            cur_pg.execute(r'''
                CALL record_thread(''' + ', '.join([r'%s']*13) + r''')
                ''', (
                id, QST_BOARD_ID, ts2dt(created_at), user_id, content,
                attachment_base, attachment_extension,
                name, email, title, misc_fields,
                None,
                updated_at,
            ))
            last = [id, not_anymore_at_least_after]

        for [
            id, created_at, user_id, content, current_reply_count,
            attachment_base, attachment_extension,
            name, email, title, misc_fields,
            latest_checked_at, is_disappeared,
            current_revision_checked_at,
        ] in conn_s3.execute(r'''
                SELECT thread.*,
                    checked_at AS latest_checked_at, is_disappeared,
                    MAX(not_anymore_at_least_after) AS current_revision_checked_at
                FROM thread
                LEFT JOIN thread_extra ON thread.id = thread_extra.id
                LEFT JOIN thread_old_revision ON thread.id = thread_old_revision.id
                GROUP BY thread.id
            '''):
            misc_fields = remove_fid_field(misc_fields)
            cur_pg.execute(r'''
                CALL record_thread(''' + ', '.join([r'%s']*13) + r''')
                ''', (
                id, QST_BOARD_ID, ts2dt(created_at), user_id, content,
                attachment_base, attachment_extension,
                name, email, title, misc_fields,
                current_reply_count,
                find_updated_at(conn_s3, id, ts2dt(
                    current_revision_checked_at)),
            ))
            cur_pg.execute(r'''
                CALL report_is_thread_disappeared(%s, %s, %s)
            ''', (id, bool(is_disappeared), ts2dt(latest_checked_at)))


def migrate_post_table(conn_s3: sqlite3.Connection, conn_pg: psycopg2._psycopg.connection):
    with conn_pg.cursor() as cur_pg:
        n = count_rows(conn_s3, 'post')
        for i, [
            id, parent_thread_id, created_at, user_id, content,
            attachment_base, attachment_extension,
            name, email, title, misc_fields,
        ] in enumerate(conn_s3.execute(r'SELECT * FROM post')):
            cur_pg.execute(r'''
                CALL record_response(''' + ', '.join([r'%s']*12) + r''')
                ''', (
                id, parent_thread_id, ts2dt(created_at), user_id, content,
                attachment_base, attachment_extension,
                name, email, title, misc_fields,
                None,
            ))
            if i % 100 == 0:
                print(f"{i+1}/{n}")


def migrate_publication_tables(conn_s3: sqlite3.Connection, conn_pg: psycopg2._psycopg.connection):
    for [
        record_id,
        _date, _type, uuid, attempts, to_thread_id,
        page_count
    ] in conn_s3.execute(r'''
        SELECT publishing_trace.*, count(published_post.trace_id)
        FROM publishing_trace
        LEFT JOIN published_post ON published_post.trace_id = publishing_trace.id
		GROUP BY publishing_trace.id
    '''):
        attempts = int(attempts)
        _date = date.fromisoformat(_date)
        assert(not PublicationRecord.is_report_published(
            conn_pg, _date, 'trend'))
        record = PublicationRecord(
            conn=conn_pg, subject_date=_date,
            report_type='trend', uuid=uuid,
        )
        for _ in range(0, attempts):
            record.increase_attempts()
        assert(attempts == record.attempts)
        record.report_thread_id_and_reply_count(to_thread_id, page_count)
        for [
            _, _,  # page_id, record_id,
            page_number, reply_post_id, reply_offset
        ] in conn_s3.execute(r'SELECT * FROM published_post WHERE trace_id = ?', (record_id,)):
            record.report_found_reply_post(
                page_number, reply_post_id, reply_offset)
        assert(record.is_done)


if __name__ == '__main__':
    main()
