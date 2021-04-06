from datetime import datetime

import sys
sys.path.append("..")  # noqa

from models.stat import ThreadStats


super_huge_thread_pg = ThreadStats(
    id=123456789,
    created_at=datetime(2006, 1, 2, 15, 4, 5, 999999),
    is_new=False,
    is_disappeared=False,

    title=None, name=None,
    raw_content="测试撑爆"*100+'\nfoo\n<font color="blue">' + "撑爆" * 100 + '</font>',

    increased_response_count=987654,
    total_reply_count=1234567,
    increased_response_count_by_po=123456,
    distinct_cookie_count=654321,
    increased_character_count=100000,
    increased_character_count_by_po=100000,

    blue_text="撑爆" * 100,
    are_blue_texts_new=True,
)
