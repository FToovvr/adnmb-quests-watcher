#!/usr/bin/env python3

from typing import Optional, List, Set, Callable

from datetime import datetime, date, timedelta
import psycopg2
from bs4 import BeautifulSoup
import jieba
import regex
from wordcloud import WordCloud
import hashlib
from random import Random

import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))  # noqa

from commons.consts import get_target_date, local_tz
from commons.config import load_config


def main():

    os.chdir(sys.path[0])

    if len(sys.argv) > 1:
        subject_date = date.fromisoformat(sys.argv[1])
    else:
        subject_date = get_target_date(datetime.now(tz=local_tz))

    print(f"日期：{subject_date.isoformat()}")

    config = load_config('../config.yaml')

    stop_words = load_default_stop_words()

    with psycopg2.connect(config.database.connection_string) as conn:
        img = generate_wordcloud(conn, subject_date, stop_words,
                                 lambda total, i: print(f"{i+1}/{total}") if i % 1000 == 0 else None)

    img.save(f'../report_out/wordcloud_{subject_date.isoformat()}.png')


def load_default_stop_words():
    files = list(filter(lambda f: os.path.join(sys.path[0], f),
                        ['./stopwords.txt', './stopwords_supplement.txt']))
    return load_stop_words_from_files(files)


def load_stop_words_from_files(stop_words_files: List[str]):
    stop_words = set()
    for f_path in stop_words_files:
        with open(f_path) as f:
            _stop_words = f.read()
            _stop_words = _stop_words.splitlines()
            _stop_words = map(lambda x: x.split('#')[0].strip(), _stop_words)
            stop_words = stop_words | set(_stop_words)
    return stop_words


def generate_wordcloud(conn: psycopg2._psycopg.connection,
                       subject_date: date, stop_words: Optional[Set[str]] = None,
                       loading_progress_callback: Optional[Callable[[int, int], None]] = None):

    if stop_words is None:
        stop_words = load_default_stop_words()

    with conn.cursor() as cur:
        cur: psycopg2._psycopg.cursor = cur

        range = (f'{subject_date.isoformat()} 04:00+8',
                 f'{(subject_date + timedelta(days=1)).isoformat()} 04:00+8')

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

            if loading_progress_callback:
                loading_progress_callback(total, i)

    def md5_color_func(word: str = None, font_size=None, position=None,
                       orientation=None, font_path=None, random_state=None):
        md5 = hashlib.md5(word.encode('utf-8')).hexdigest()
        x = int(md5[:6], base=16) / float(16**6 - 1) * 240
        return f'hsl({x}, 80%, 50%)'

    random_state = Random()
    random_state.seed("( ﾟ∀。)")
    wc = WordCloud(
        random_state=random_state,
        background_color='white',
        color_func=md5_color_func,
        font_path='./fonts/NotoSerifSC/NotoSerifSC-SemiBold.otf',
        width=1920, height=1200,
        scale=2,
    ).generate(' '.join(words))

    # 这文件夹终于有作用了…
    return wc.to_image()


if __name__ == '__main__':
    main()
