from contextlib import contextmanager
from os import environ

import pytest
from testcontainers.postgres import PostgresContainer

from citatio.db import PostgreSQLDatabase


pytestmark = pytest.mark.postgresql


@contextmanager
def local_pgvector_container():
    with PostgresContainer('pgvector/pgvector:pg18', driver=None) as container:
        yield container


@pytest.fixture(scope='session')
def connect_pgvector():
    match environ:
        case {'GITHUB_ACTIONS': _}:
            # running on GHA, provide connection details to pgvector service defined in .github/workflows/citatio.yml
            yield {
                'host': environ.get('POSTGRES_HOST', 'postgres'),
                'port': environ.get('POSTGRES_PORT', 5432),
                'user': environ.get('POSTGRES_USER', 'postgres'),
                'password': environ.get('POSTGRES_PASSWORD'),
                'database': environ.get('POSTGRES_DATABASE', 'postgres'),
            }
        case _:
            # running locally, provide connection details to local pgvector container
            with local_pgvector_container() as container:
                yield {
                    'host': container.get_container_host_ip(),
                    'port': container.port,
                    'user': container.username,
                    'password': container.password,
                    'database': container.dbname,
                }


@pytest.fixture(scope='session')
def database(connect_pgvector):
    return PostgreSQLDatabase(**connect_pgvector)


async def test_auto_create_schema_idempotent(connect_pgvector, database):
    select_tables = 'SELECT table_name FROM information_schema.tables'
    tables = {name.lower() for name in await database.connection.fetch(select_tables)}
    expected = {'functions', 'labels'}
    assert tables & expected == expected

    database2 = PostgreSQLDatabase(**connect_pgvector)
    assert database != database2
    assert database.connection != database2.connection
    assert tables == {name.lower() for name in await database2.connection.fetch(select_tables)}
