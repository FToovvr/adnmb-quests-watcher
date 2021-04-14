#!/usr/bin/env python3

from typing import Optional, Dict, List, Set, Callable
from collections import defaultdict, Counter
from operator import add

from functools import reduce
from datetime import datetime, date, timedelta
import hashlib
from random import Random
from pathlib import Path
import json
from statistics import mean, stdev, quantiles
import math

import psycopg2
from bs4 import BeautifulSoup
# import jieba
import pkuseg
import regex
from wordcloud import WordCloud

import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))  # noqa

from commons.consts import get_target_date, local_tz
from commons.config import load_config


seg = pkuseg.pkuseg()

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
    img_path = Path(f'../report_out/wordcloud_{subject_date.isoformat()}.png')
    if img_path.exists():
        img_path.unlink()
    img.save(img_path)


def load_default_stop_words():
    files = list(filter(lambda f: os.path.join(sys.path[0], f), [
                            './stopwords.txt',
                            # './stopwords_supplement.txt',
                            ]))
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

    dc = update_dc(conn, subject_date)

    words = []
    total = total_contents(conn, subject_date)
    for i, [_, content] in enumerate(each_content_on(conn, subject_date)):
        words += segment_content(content)
        if loading_progress_callback:
            loading_progress_callback(total, i)

    data_count = dc['count']
    words = Counter(words)
    total_words_today = sum(words.values())
    max_tf_today = words.most_common(1)[0][1] / total_words_today

    stop_words = load_default_stop_words()

    def calculate_tf_idf(word, count):
        def tf(count):
            return count / total_words_today
        if word in stop_words or word in ['No']:
            return 0
        tf_idf = (0.1 + 0.9*(tf(count) / max_tf_today)) * \
            math.log10(dc['n']/(data_count[word]+1))
        return tf_idf
    tf_idfs = {word: calculate_tf_idf(word, count)
               for word, count in words.items()}

    with open(os.path.join(sys.path[0], '../report_out', f'tf-idf_{subject_date.isoformat()}.txt'), 'w+') as xf:
        for [word, value] in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True):
            xf.write(f'{word} {value}\n')

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
        width=800, height=600,
        scale=2,
    ).generate_from_frequencies(tf_idfs)

    # 这文件夹终于有作用了…
    return wc.to_image()


class NextDict(dict):

    def __init__(self, orignal_next_dict: Dict[str, Dict[str, int]], *nargs, **kwargs):
        super(NextDict, self).__init__(*nargs, **kwargs)
        self.orignal_next_dict = orignal_next_dict

    def __missing__(self, key):
        if key in self.orignal_next_dict:
            self[key] = defaultdict(int, **self.orignal_next_dict[key])
        else:
            self[key] = defaultdict(int)
        return self[key]


def update_dc(conn: psycopg2._psycopg.connection, subject_date: date):
    """dc = data count"""
    dc_file_path = os.path.join(sys.path[0], 'dc.json')
    if not Path(dc_file_path).exists():
        dc = {}
        # 有多少组
        dc['n'] = 0
        # 如果以天、主串分组，某个词一共出现在几组过
        dc['count'] = defaultdict(int)
        # 某个词后面跟了哪些词多少次
        dc['next'] = NextDict({})
        with conn.cursor() as cur:
            cur: psycopg2._psycopg.cursor = cur
            cur.execute(r'''
                SELECT run_at FROM activity
                ORDER BY run_at ASC
                LIMIT 1
            ''')
            first_created_at: datetime = cur.fetchone()[0]
            if first_created_at.hour < 4:
                first_created_at = first_created_at.date() - timedelta(days=1)
            else:
                first_created_at = first_created_at.date()
            # 第一天大概率不全
            start_date = first_created_at + timedelta(days=1)
    else:
        with open(dc_file_path, 'r') as f:
            dc = json.load(f)
            start_date = date.fromisoformat(
                dc['last_updated_date']) + timedelta(days=1)
            dc['count'] = defaultdict(int, **dc['count'])
            _next_items = dc['next']
            dc['next'] = NextDict(_next_items)

    print(f"dc start date: {start_date.isoformat()}")

    updated = False
    if subject_date >= start_date:
        # https://stackoverflow.com/a/24637447
        for current_date in [start_date + timedelta(days=x) for x in range(0, (subject_date-start_date).days + 1)]:
            updated = True
            total = total_contents(conn, current_date)
            dc['n'] += total
            seen_words_per_thread = defaultdict(set)
            for i, [thread_id, content] in enumerate(each_content_on(conn, current_date)):
                if i % 1000 == 0:
                    print(f'dc {current_date.isoformat()} {i+1}/{total}')
                words = segment_content(content)
                for i, word in enumerate(words):
                    seen_words_per_thread[thread_id].add(word)
                    if i < len(words) - 1:
                        dc['next'][word][words[i+1]] += 1
            counts = Counter(
                reduce(add, map(lambda x: list(x), seen_words_per_thread.values())))
            dc['count'] = Counter(dc['count']) + counts

        dc['last_updated_date'] = subject_date.isoformat()

    if updated:
        with open(dc_file_path, 'w+') as f:
            json.dump(dc, f, sort_keys=True, indent=4, ensure_ascii=False)

    dc['count'] = Counter(**dc['count'])

    return dc


def find_outlier(x: Dict[str, int]):
    l = list(x)
    values = map(lambda item: item[1], l)
    # mean = mean(values)
    # stdev = stdev(values)
    quartiles = quantiles(values)
    iqr = quartiles[2] - quartiles[0]

    # https://stackoverflow.com/a/2303583
    # outliers = filter(lambda item: abs(item[1]-mean) > 3 * stdev, x)
    outliers = filter(lambda item: item[0] > quartiles[2] + iqr * 1.5)

    return list(map(lambda item: item[0], outliers))


def segment_content(content: str):
    content = BeautifulSoup(content, features='html.parser').get_text()
    text = ' '.join(regex.findall(
        r'[\p{Han}]+|[\p{Latin}][\p{Latin}-]*', content))
    if len(text.strip()) == 0:
        return []
    # words += filter(lambda w: w not in stop_words, jieba.lcut(text))
    # return jieba.lcut(text)
    return seg.cut(text)


def total_contents(conn: psycopg2._psycopg.connection, subject_date: date):
    with conn.cursor() as cur:
        cur.execute(
            r'SELECT count(id) FROM post WHERE in_boundaries(created_at, %s::timestamptz, %s::timestamptz)', get_range(subject_date))
        return cur.fetchone()[0]


def each_content_on(conn: psycopg2._psycopg.connection, subject_date: date):
    with conn.cursor() as cur:
        cur.execute(r'SELECT id, parent_thread_id,  content FROM post WHERE in_boundaries(created_at, %s::timestamptz, %s::timestamptz)',
                    get_range(subject_date))
        for [id, parent_thread_id, content] in cur:
            yield [parent_thread_id or id, content]


def get_range(subject_date: date):
    return (f'{subject_date.isoformat()} 04:00+8',
            f'{(subject_date + timedelta(days=1)).isoformat()} 04:00+8')


if __name__ == '__main__':
    main()
