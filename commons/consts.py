from datetime import datetime, timedelta, date

from dateutil import tz

local_tz = tz.gettz('Asia/Shanghai')

ZWSP = '\u200b'

OMITTING = ZWSP + "…"


def get_target_date(now: datetime = None) -> date:
    if now is None:
        now = datetime.now(tz=local_tz)
    return (now - timedelta(hours=4)).date() - timedelta(days=1)
