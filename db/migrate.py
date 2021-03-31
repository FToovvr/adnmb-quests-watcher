#!/usr/bin/env python3

import sys

import sqlite3
import psycopg2

from datetime import datetime, date
from dateutil import tz

local_tz = tz.gettz('Asia/Shanghai')


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
        with conn_pg.cursor() as cur_pg:

            for migrate_fn in [
                migrate_activity_table,
                migrate_thread_table,
                migrate_thread_old_revision_table,
                migrate_thread_extra_table,
                migrate_post_table,
                migrate_completion_registry_entry,
                migrate_publishing_trace,
                migrate_published_post,
            ]:
                print(f"start: {migrate_fn.__name__}")
                migrate_fn(conn_s3, cur_pg)
                print(f"done: {migrate_fn.__name__}")

    conn_pg.close()
    conn_s3.close()


def migrate_activity_table(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
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


def migrate_thread_table(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, created_at, user_id, content, current_reply_count,
        attachment_base, attachment_extension,
        name, email, title, misc_fields,
    ] in conn_s3.execute(r'SELECT * FROM thread'):
        cur_pg.execute(r'''
            INSERT INTO thread
            VALUES (''' + ', '.join([r'%s']*11) + r''')
            ON CONFLICT DO NOTHING
            ''', (
            id, ts2dt(created_at), user_id, content,
            attachment_base, attachment_extension,
            name, email, title, misc_fields,
            current_reply_count,
        ))


def migrate_thread_old_revision_table(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, not_anymore_at_least_after, content, name, email, title,
    ] in conn_s3.execute(r'SELECT * FROM thread_old_revision'):
        cur_pg.execute(r'''
            INSERT INTO thread_old_revision (id, not_anymore_at_least_after, content, name, email, title, is_not_complete)
            VALUES (''' + ', '.join([r'%s']*7) + r''')
            ON CONFLICT DO NOTHING
        ''', (
            id, ts2dt(
                not_anymore_at_least_after), content, name, email, title, True,
        ))


def migrate_thread_extra_table(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, checked_at, is_disappeared,
    ] in conn_s3.execute(r'SELECT * FROM thread_extra'):
        cur_pg.execute(r'''
            INSERT INTO thread_extra
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        ''', (id, ts2dt(checked_at), bool(is_disappeared)))


def migrate_post_table(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    n = count_rows(conn_s3, 'post')
    for i, [
        id, parent_thread_id, created_at, user_id, content,
        attachment_base, attachment_extension,
        name, email, title, misc_fields,
    ] in enumerate(conn_s3.execute(r'SELECT * FROM post')):
        cur_pg.execute(r'''
            INSERT INTO post
            VALUES (''' + ', '.join([r'%s']*11) + r''')
            ON CONFLICT DO NOTHING
            ''', (
            id, ts2dt(created_at), user_id, content,
            attachment_base, attachment_extension,
            name, email, title, misc_fields,
            parent_thread_id,
        ))
        if i % 100 == 0:
            print(f"{i+1}/{n}")


def migrate_completion_registry_entry(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, post_id, subject_thraed_id, has_blue_text_been_added,
    ] in conn_s3.execute(r'SELECT * FROM completion_registry_entry'):
        cur_pg.execute(r'''
            INSERT INTO completion_registry_entry
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        ''', (id, post_id, subject_thraed_id, bool(has_blue_text_been_added)))


def migrate_publishing_trace(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, _date, _type, uuid, attempts, to_thread_id
    ] in conn_s3.execute(r'SELECT * FROM publishing_trace'):
        cur_pg.execute(r'''
            INSERT INTO publishing_trace
            VALUES (''' + ', '.join([r'%s']*6) + r''')
            ON CONFLICT DO NOTHING
            ''', (
            id, date.fromisoformat(_date), _type, uuid, attempts, to_thread_id,
        ))


def migrate_published_post(conn_s3: sqlite3.Connection, cur_pg: psycopg2._psycopg.cursor):
    for [
        id, trace_id, page_number, reply_post_id, reply_offset,
    ] in conn_s3.execute(r'SELECT * FROM published_post'):
        cur_pg.execute(r'''
            INSERT INTO published_post
            VALUES (''' + ', '.join([r'%s']*5) + r''')
            ON CONFLICT DO NOTHING
            ''', (
            id, trace_id, page_number, reply_post_id, reply_offset,
        ))


if __name__ == '__main__':
    main()
