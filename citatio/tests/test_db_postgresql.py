from contextlib import asynccontextmanager
from itertools import pairwise
from os import environ

import numpy
import pytest
from numpy.linalg import norm
from testcontainers.postgres import PostgresContainer

from citatio.db import PostgreSQLDatabase


pytestmark = pytest.mark.postgresql


@asynccontextmanager
async def local_pgvector_container():
    with PostgresContainer('pgvector/pgvector:pg18', driver=None) as container:
        yield container


@pytest.fixture(scope='session')
async def connect_pgvector():
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
            async with local_pgvector_container() as container:
                yield {
                    'host': container.get_container_host_ip(),
                    'port': container.port,
                    'user': container.username,
                    'password': container.password,
                    'database': container.dbname,
                }


@pytest.fixture(scope='session')
async def database(connect_pgvector):
    return await PostgreSQLDatabase.connect(**connect_pgvector)


@pytest.fixture(scope='function')
async def empty_database(database):
    await database.connection.execute("""
        DELETE FROM labels;
        DELETE FROM functions;
    """)
    yield database


@pytest.fixture(scope='function')
async def filled_database(empty_database, functions, embeddings):
    for function in functions:
        await empty_database.add_function(
            function['name'],
            function['cfg'],
            embeddings[function['name']],
            function['binary_name'],
            function['binary_sha256'],
        )

    yield empty_database


async def test_auto_create_schema_idempotent(connect_pgvector, database):
    select_tables = 'SELECT table_name FROM information_schema.tables'
    tables = {name.get('table_name').lower() for name in await database.connection.fetch(select_tables)}
    expected = {'functions', 'labels'}
    assert tables & expected == expected

    database2 = await PostgreSQLDatabase.connect(**connect_pgvector)
    assert database != database2
    assert database.connection != database2.connection
    assert tables == {name.get('table_name').lower() for name in await database2.connection.fetch(select_tables)}


async def test_add_duplicate(empty_database, functions, embeddings):
    function = functions[0]
    embedding = embeddings[function['name']]

    function_id = await empty_database.add_function(
        function['name'],
        function['cfg'],
        embedding,
        function['binary_name'],
        function['binary_sha256'],
    )

    assert (
        await empty_database.add_function(
            function['name'],
            function['cfg'],
            embedding,
            function['binary_name'],
            function['binary_sha256'],
        )
        == function_id
    )


async def test_search_identical(filled_database, embeddings):
    results = await filled_database.search_function(embeddings['init_have_lse_atomics'])

    assert results[0]['function'] == 'init_have_lse_atomics'
    # similarity for that function should be 1.0
    assert results[0]['similarity'] == pytest.approx(1.0)
    # similarities should be declining throughout the results
    assert all(a['similarity'] >= b['similarity'] for a, b in pairwise(results))


async def test_search_similar(filled_database, embeddings):
    jitter = numpy.random.rand(768).astype(dtype=numpy.float32) / 1_000.0
    query_embedding = embeddings['thunk_1234'] + jitter
    # normally, embeddings should have been normalized by the model, make sure to manually normalize the query here
    query_embedding = query_embedding / norm(query_embedding)
    results = await filled_database.search_function(query_embedding)

    assert results[0]['function'] == 'thunk_1234'
    assert 0.9 < results[0]['similarity'] < 1.0


async def test_search_top_1(filled_database, embeddings):
    results = await filled_database.search_function(embeddings['init_have_lse_atomics'], top_n=1)

    assert len(results) == 1


async def test_search_duplicate_label(filled_database, functions, embeddings):
    _init = next(function for function in functions if function['name'] == '_init')
    # add the same function with the same label as if it were from a different binary
    await filled_database.add_function(
        _init['name'], _init['cfg'], embeddings[_init['name']], 'another_binary', '1234abcd' * 16
    )

    results = await filled_database.search_function(embeddings[_init['name']], top_n=2)

    assert len(results) == 2
    assert '_init' == results[0]['function'] == results[1]['function']
    assert results[0]['binary_name'] != results[1]['binary_name']
