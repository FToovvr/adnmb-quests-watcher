from typing import Optional, List
from dataclasses import dataclass

from pathlib import Path
from os.path import join
import re

import yaml

import anobbsclient


@dataclass(frozen=True)
class ClientConfig:

    host: str

    client_user_agent: str
    client_appid: Optional[str]

    user_userhash: str

    def create_client(self) -> anobbsclient.Client:
        return anobbsclient.Client(
            user_agent=self.client_user_agent,
            host=self.host,
            appid=self.client_appid,
            default_request_options={
                'user_cookie': anobbsclient.UserCookie(userhash=self.user_userhash),
                'login_policy': 'when_required',
                'gatekeeper_page_number': 100,
            },
        )


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    dbname: str
    user: str
    password: str

    @property
    def connection_string(self) -> str:
        # XXX: 也不知道够不够安全，不过反正也是自用
        for x in [self.host, self.dbname, self.user, self.password]:
            assert(re.search(r'\s', x) is None)
        return f'dbname={self.dbname} user={self.user} password={self.password} host={self.host}'


@dataclass(frozen=True)
class PublishingConfig:
    page_capacity: int
    including: List[str]


@dataclass(frozen=True)
class Config:
    board_id: int
    trend_thread_id: Optional[int]
    daily_qst_thread_id: Optional[int]
    completion_registry_thread_id: int

    database: DatabaseConfig
    client: Optional[ClientConfig]
    publishing: PublishingConfig


def load_config(path: str) -> Config:
    path = Path(path)

    with open(path, 'r') as config_file:
        obj = yaml.load(config_file.read(), Loader=yaml.SafeLoader)

        consts = obj['consts']

        database = obj['database']
        with open(join(path.parent, database['password-file']), 'r') as pw_file:
            database_password = pw_file.read().strip()
        database = DatabaseConfig(
            host=database['host'],
            dbname=database['dbname'],
            user=database['user'],
            password=database_password,
        )

        client = obj['client']
        if client is not None:
            with open(join(path.parent, client['file']), 'r') as client_file:
                client_obj = yaml.load(
                    client_file.read(), Loader=yaml.SafeLoader)
                client = ClientConfig(
                    host=client_obj['host'],
                    client_user_agent=client_obj['client']['user-agent'],
                    client_appid=client_obj['client']['appid'],
                    user_userhash=client_obj['user']['userhash'],
                )

        publishing = obj['publishing']
        publishing = PublishingConfig(
            page_capacity=publishing['page-capacity'],
            including=publishing['including'],
        )

        return Config(
            board_id=consts['board-id'],
            trend_thread_id=consts['trend-thread-id'],
            daily_qst_thread_id=consts['daily-qst-thread-id'],
            completion_registry_thread_id=consts['completion-registry-thread-id'],

            database=database,
            client=client,
            publishing=publishing,
        )
