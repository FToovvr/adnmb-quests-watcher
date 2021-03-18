import os

import anobbsclient


def get_client():
    return anobbsclient.Client(
        user_agent=os.environ['ANOBBS_CLIENT_ENVIRON'],
        host=os.environ['ANOBBS_HOST'],
        appid=os.environ.get('ANOBBS_CLIENT_APPID', None),
        default_request_options={
            'user_cookie': anobbsclient.UserCookie(userhash=os.environ['ANOBBS_USERHASH']),
            'login_policy': 'when_required',
            'gatekeeper_page_number': 100,
        },
    )
