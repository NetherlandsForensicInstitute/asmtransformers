import json
from os import environ
from pathlib import Path

import asyncpg
import numpy as np
import pytest
from confidence import Configuration
from testcontainers.postgres import PostgresContainer

from citatio.db import PostgreSQLDatabase, SQLiteDatabase


TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope='session')
def functions():
    return json.loads((TEST_ROOT / 'functions.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def embeddings(functions):
    arrays = np.load(TEST_ROOT / 'functions.npy', allow_pickle=False)
    return {function['name']: embedding for function, embedding in zip(functions, arrays, strict=True)}


@pytest.fixture(scope='session')
def local_pgvector_container():
    with PostgresContainer('pgvector/pgvector:pg18', driver=None) as container:
        yield container


@pytest.fixture
def connect_pgvector(request, monkeypatch):
    # translate PostgreSQL connection details into environment variables that will be picked up by confidence.load_name
    # in the app's lifecycle
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
            container = request.getfixturevalue('local_pgvector_container')
            yield {
                'host': container.get_container_host_ip(),
                'port': container.get_exposed_port(5432),
                'user': container.username,
                'password': container.password,
                'database': container.dbname,
            }


@pytest.fixture(
    params=[
        pytest.param('sqlite', marks=pytest.mark.sqlite),
        pytest.param('postgresql', marks=pytest.mark.postgresql),
    ]
)
async def database_config(request):
    match request.param:
        case 'sqlite':
            # use a in-memory sqlite database
            yield Configuration({'database.sqlite': ':memory:'})
        case 'postgresql':
            # run a session-scope ephemeral database
            connect = request.getfixturevalue('connect_pgvector')
            yield Configuration({'database': connect})
            # empty the database after use
            connection = await asyncpg.connect(**connect)
            await connection.execute("""
                DROP TABLE IF EXISTS labels CASCADE;
                DROP TABLE IF EXISTS functions CASCADE;
            """)
            await connection.close()


@pytest.fixture
async def database_env(monkeypatch, database_config):
    match database_config:
        case {'database.sqlite': fname}:
            # NB: add quotes to avoid the configuration's format misparsing :memory:
            monkeypatch.setenv('CITATIO_DATABASE_SQLITE', f'"{fname}"')
            yield
        case {'database': connect}:
            for var, value in connect.items():
                monkeypatch.setenv(f'CITATIO_DATABASE_{var}'.upper(), str(value))
            yield
        case _:
            raise ValueError


@pytest.fixture
async def database(database_config):
    match database_config:
        case {'database.sqlite': name}:
            # create an in-memory database that will be empty after use by design
            async with await SQLiteDatabase.connect(name) as db:
                yield db
        case {'database': connect}:
            async with await PostgreSQLDatabase.connect(**connect) as db:
                yield db
        case _:
            raise ValueError
