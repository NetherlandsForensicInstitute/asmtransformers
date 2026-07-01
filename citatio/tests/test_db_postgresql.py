from contextlib import contextmanager
from os import environ

import pytest
from testcontainers.postgres import PostgresContainer

from citatio.db import PostgreSQLDatabase


pytestmark = pytest.mark.postgresql


@contextmanager
def _local_pgvector():
    with PostgresContainer('pgvector/pgvector:pg18', driver=None) as container:
        yield container


@pytest.fixture(scope='session')
def pgvector():
    match environ:
        case {'GITHUB_ACTIONS': _}:
            yield {
                'host': environ.get('POSTGRES_HOST', 'postgres'),
                'port': environ.get('POSTGRES_PORT', 5432),
                'user': environ.get('POSTGRES_USER', 'postgres'),
                'password': environ.get('POSTGRES_PASSWORD'),
                'database': environ.get('POSTGRES_DATABASE', 'postgres'),
            }
        case _:
            with _local_pgvector() as container:
                yield {
                    'host': container.get_container_host_ip(),
                    'port': container.port,
                    'user': container.username,
                    'password': container.password,
                    'database': container.dbname,
                }


def test_create_schema(pgvector):
    database = PostgreSQLDatabase(**pgvector)
    assert database
