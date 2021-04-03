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
    cur.execute(r'SELECT content FROM post')
    for i, [content] in enumerate(cur.fetchall()):
        if i % 100 == 0:
            print(i+1)
        with conn.cursor() as cur2:
            cur2.execute(
                r'''SELECT remove_all_spaces, length FROM remove_all_spaces(extract_text(%s)), length(remove_all_spaces)''', (content,))
            [no_spaces_content_pg, length_pg] = cur2.fetchone()
        no_spaces_content_bs = re.sub(
            r'\s', '', BeautifulSoup(content, 'html.parser').get_text())
        if no_spaces_content_bs != no_spaces_content_pg:
            print([i+1, 'content', content,
                  no_spaces_content_bs, no_spaces_content_pg])
            x += 1
        elif len(no_spaces_content_bs) != length_pg:
            print([i+1, 'length', content,
                   len(no_spaces_content_bs), length_pg])
        if x == 10:
            break


if __name__ == '__main__':
    main()
