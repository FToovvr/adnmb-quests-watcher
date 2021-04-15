#!/usr/bin/env python3

from typing import Optional, Dict, List, Set, Callable
from dataclasses import dataclass
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
import jieba
# import pkuseg
import regex
from wordcloud import WordCloud

import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))  # noqa

from commons.consts import get_target_date, local_tz
from commons.config import load_config


# seg = pkuseg.pkuseg()


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
    stop_words = set([' '])
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
    stop_words = load_default_stop_words()
    adjust_dc(dc, stop_words)
    data_count = dc['count']

    words = []
    total = total_contents(conn, subject_date)
    for i, [_, content] in enumerate(each_content_on(conn, subject_date)):
        words += filter(lambda word: word in data_count,
                        segment_content(content))
        if loading_progress_callback:
            loading_progress_callback(total, i)

    words = Counter(words)
    total_words_today = sum(words.values())
    max_tf_today = words.most_common(1)[0][1] / total_words_today

    def calculate_tf_idf(word, count):
        def tf(count):
            return count / total_words_today
        if word in stop_words or word in ['No']:
            return 0
        # print(word, count, tf(count), max_tf_today, dc['n'], data_count[word])
        tf_idf = (0.5 + 0.5*(tf(count) / max_tf_today)) * \
            math.log10(dc['n']/(data_count[word]))
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


@dataclass
class NextDict():

    _dict: Dict[str, Dict[str, int]]

    def __init__(self, orignal_next_dict: Dict[str, Dict[str, int]], *nargs, **kwargs):
        self._dict = orignal_next_dict

    def __getitem__(self, key):
        s = super(NextDict, self)
        if key in self._dict:
            if not isinstance(self._dict[key], defaultdict):
                self._dict[key] = defaultdict(int, **self._dict[key])
        else:
            self._dict[key] = defaultdict(int)
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def items(self):
        return self._dict.items()

    def get_dict(self):
        return self._dict


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
            dc['next'] = NextDict(dc['next'])

    print(f"dc start date: {start_date.isoformat()}")

    updated = False
    if subject_date >= start_date:
        # https://stackoverflow.com/a/24637447
        for current_date in [start_date + timedelta(days=x) for x in range(0, (subject_date-start_date).days + 1)]:
            updated = True
            total = total_contents(conn, current_date)
            # dc['n'] += total
            dc['n'] += 1
            # seen_words_per_thread = defaultdict(set)
            seen_words_today = set()
            for i, [thread_id, content] in enumerate(each_content_on(conn, current_date)):
                if i % 1000 == 0:
                    print(f'dc {current_date.isoformat()} {i+1}/{total}')
                words = segment_content(content)
                for i, word in enumerate(words):
                    if word == ' ':
                        continue
                    # seen_words_per_thread[thread_id].add(word)
                    seen_words_today.add(word)
                    if i < len(words) - 1 and words[i+1] != ' ':
                        dc['next'][word][words[i+1]] += 1
                    dc['next'][word]['$total'] += 1
            # counts = Counter(
            #     reduce(add, map(lambda x: list(x), seen_words_per_thread.values())))
            counts = Counter(list(seen_words_today))
            dc['count'] = Counter(dc['count']) + counts

        dc['last_updated_date'] = subject_date.isoformat()

    if updated:
        def i_f___ing_hate_python(obj):
            if isinstance(obj, NextDict):
                return obj.get_dict()
            return obj.__dict__
        with open(dc_file_path, 'w+') as f:
            json.dump(dc, f, sort_keys=True, indent=4,
                      ensure_ascii=False, default=i_f___ing_hate_python)

    dc['count'] = Counter(**dc['count'])

    return dc


def adjust_dc(dc, stop_words: Optional[set] = None):
    if stop_words is None:
        stop_words = load_default_stop_words()

    dc_count = dc['count']
    for stop_word in stop_words:
        dc_count.pop(stop_word, None)
    dc_count_orig = dict(**dc_count)

    for word, next in dc['next'].items():
        if word in stop_words:
            continue
        outliers = find_outliers(next)
        outliers = list(filter(lambda x: x[0] not in stop_words, outliers))
        # if len(outliers) > 0:
        #     print(word, outliers)
        for [outlier_word, outliers_count] in outliers:
            dc_count[word+outlier_word] += dc_count_orig[word] * \
                (outliers_count/next['$total'])
            for x in [word, outlier_word]:
                dc_count[x] = max(0, dc_count[x] - dc_count_orig[x]
                                  * (outliers_count/dc['next'][x]['$total']))
                if dc_count[x] == 0:
                    dc_count.pop(x)
                elif int(dc_count[x]) == 0:
                    # workaround
                    dc_count[x] = 1

    dc['count'] = dc_count


def find_outliers(x: Dict[str, int]):
    l = list(filter(lambda item: not item[0].startswith('$'), x.items()))
    if len(l) < 3:
        return []
    values = list(map(lambda item: item[1], l))
    m = mean(values)
    # if m < 10:
    #     return []
    s = stdev(values)
    # quartiles = quantiles(values)
    # iqr = quartiles[2] - quartiles[0]

    # https://stackoverflow.com/a/2303583
    outliers = filter(lambda item: abs(item[1]-m) > 3*2 * s, l)
    # outliers = filter(lambda item: item[1] > quartiles[2] + iqr * 3, l)

    return list(outliers)  # list(map(lambda item: item[0], outliers))


def segment_content(content: str):
    content = BeautifulSoup(content, features='html.parser').get_text()
    text = ' '.join(regex.findall(
        r'[\p{Han}]+|[\p{Latin}][\p{Latin}-]*', content))
    if len(text.strip()) == 0:
        return []
    # words += filter(lambda w: w not in stop_words, jieba.lcut(text))
    return jieba.lcut(text)
    # return seg.cut(text)


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
