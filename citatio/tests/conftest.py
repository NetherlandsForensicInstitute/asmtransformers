import json
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from testcontainers.postgres import PostgresContainer

from citatio.db import SQLiteDatabase


TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope='session')
def monkeypatch_session():
    # regular monkeypatch fixture is function scope, provide a session-scoped one
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope='session')
def functions():
    return json.loads((TEST_ROOT / 'functions.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def embeddings(functions):
    arrays = np.load(TEST_ROOT / 'functions.npy', allow_pickle=False)
    return {function['name']: embedding for function, embedding in zip(functions, arrays, strict=True)}


@asynccontextmanager
async def local_pgvector_container():
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
async def connect_pgvector(monkeypatch_session):
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
            async with local_pgvector_container() as container:
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


@pytest.fixture(
    params=[
        pytest.param('sqlite', marks=pytest.mark.sqlite),
    ]
)
async def database(request):
    match request.param:
        case 'sqlite':
            yield await SQLiteDatabase.from_name(':memory:')
        case _:
            raise ValueError
