#!/usr/bin/env python3

import re

import psycopg2
from bs4 import BeautifulSoup


def main():
    with open('password.secret', 'r') as password_file:
        password = password_file.read().strip()
    conn: psycopg2._psycopg.connection = psycopg2.connect(
        f'dbname=adnmb_qst_watcher user=postgres password={password} host=pi')
    cur: psycopg2._psycopg.cursor = conn.cursor()

    x = 0
    cur.execute(r'SELECT id, content FROM post')
    for i, [id, content] in enumerate(cur):
        if i % 1000 == 0:
            print(i+1)
        with conn.cursor() as cur2:
            cur2.execute(
                r'''SELECT count_content_characters_works(%s)''', (content,))
            [length_pg] = cur2.fetchone()
        no_spaces_content_bs = re.sub(
            r'\s', '', BeautifulSoup(content, 'html.parser').get_text())
        if len(no_spaces_content_bs) != length_pg:
            print([i+1, x+1, 'length', len(no_spaces_content_bs), length_pg,
                   id, content])
            x += 1
        if x == 10:
            break


if __name__ == '__main__':
    main()
