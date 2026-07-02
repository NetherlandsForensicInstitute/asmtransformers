import os
from contextlib import contextmanager
from os import environ

import asyncpg
import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from citatio.api import app


pytestmark = pytest.mark.postgresql


@contextmanager
def local_pgvector_container():
    with PostgresContainer('pgvector/pgvector:pg18', driver=None) as container:
        yield container


def _patch_database_env(monkeypatch, env):
    for var, value in env.items():
        # use environment variables to communicate configuration
        var = f'CITATIO_DATABASE_{var}'.upper()
        if value:
            monkeypatch.setenv(var, str(value))
        else:
            # env vars have no concept of 'no value' other than removing the var
            monkeypatch.delenv(var)

    # return the original env that's still suitable for use with asyncpg.connect()
    return env


@pytest.fixture(scope='session')
def connect_pgvector(monkeypatch_session):
    # translate PostgreSQL connection details into environment variables that will be picked up by confidence.load_name
    # in the app's lifecycle
    match environ:
        case {'GITHUB_ACTIONS': _}:
            # running on GHA, provide connection details to pgvector service defined in .github/workflows/citatio.yml
            yield _patch_database_env(
                monkeypatch_session,
                {
                    'host': environ.get('POSTGRES_HOST', 'postgres'),
                    'port': environ.get('POSTGRES_PORT', 5432),
                    'user': environ.get('POSTGRES_USER', 'postgres'),
                    'password': environ.get('POSTGRES_PASSWORD'),
                    'database': environ.get('POSTGRES_DATABASE', 'postgres'),
                },
            )
        case _:
            # running locally, provide connection details to local pgvector container
            with local_pgvector_container() as container:
                yield _patch_database_env(
                    monkeypatch_session,
                    {
                        'host': container.get_container_host_ip(),
                        'port': container.get_exposed_port(5432),
                        'user': container.username,
                        'password': container.password,
                        'database': container.dbname,
                    },
                )


@pytest.fixture
async def client(connect_pgvector):
    with TestClient(app) as client:
        yield client

    # after the test (post-yield), make sure to empty the tables that we might have inserted into during the test
    connection = await asyncpg.connect(**connect_pgvector)
    await connection.execute("""
        DELETE FROM labels;
        DELETE FROM functions;
    """)


@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="don't run this test on CI")
def test_add_function(client, functions):
    response = client.post('/api/v1/add', json=functions[0])
    assert response.status_code == 200


@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="don't run this test on CI")
def test_search_known(client, functions):
    for function in functions:
        client.post('/api/v1/add', json=function)

    for function in functions:
        results = client.post('/api/v1/search', json={'cfg': function['cfg'], 'top_n': 2}).json()
        assert len(results) == 2
        assert results[0]['similarity'] == pytest.approx(1.0)


@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="don't run this test on CI")
def test_search_unknown(client, functions):
    for function in functions[1:]:
        client.post('/api/v1/add', json=function)

    results = client.post('/api/v1/search', json={'cfg': functions[0]['cfg']}).json()
    assert len(results) == 3
    for result in results:
        # nothing matches exactly, nothing should come back < 0.0
        assert 0.1 < result['similarity'] < 0.9
