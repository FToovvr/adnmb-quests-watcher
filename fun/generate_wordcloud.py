#!/usr/bin/env python3

# 要在父文件夹运行

import psycopg2
from bs4 import BeautifulSoup
import jieba
import regex
from wordcloud import WordCloud

import sys
sys.path.append(".")  # noqa

from commons.config import load_config


def main():
    config = load_config('./config.yaml')

    stop_words = set()

    for f_path in ['fun/stopwords.txt', 'fun/stopwords_supplement.txt']:
        with open(f_path) as f:
            _stop_words = f.read()
            _stop_words = _stop_words.splitlines()
            _stop_words = map(lambda x: x.split('#')[0].strip(), _stop_words)
            stop_words = stop_words | set(_stop_words)

    conn = psycopg2.connect(config.database.connection_string)
    cur: psycopg2._psycopg.cursor = conn.cursor()

    range = ('2021-04-11 04:00+8', '2021-04-12 04:00+8')

    cur.execute(
        r'SELECT count(id) FROM post WHERE in_boundaries(created_at, %s::timestamptz, %s::timestamptz)', range)
    total = cur.fetchone()[0]

    cur.execute(r'SELECT content FROM post WHERE in_boundaries(created_at, %s::timestamptz, %s::timestamptz)',
                range)

    words = []

    for i, [content] in enumerate(cur):
        content = BeautifulSoup(content, features='html.parser').get_text()
        text = ' '.join(regex.findall(
            r'[\p{Han}]+|[\p{Latin}][\p{Latin}-]*', content))
        words += filter(lambda w: w not in stop_words, jieba.lcut(text))

        if i % 1000 == 0:
            print(f"{i+1}/{total}")

    wc = WordCloud(
        # 这文件夹终于有作用了…
        font_path='./fun/fonts/NotoSerifSC/NotoSerifSC-SemiBold.otf',
        width=1920, height=1200,
        background_color='white',
        scale=2,
    ).generate(' '.join(words))

    wc.to_file('./report_out/wordcloud.png')


if __name__ == '__main__':
    main()
