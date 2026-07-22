from itertools import pairwise

import numpy
import pytest

from citatio.db import PostgreSQLDatabase, SQLiteDatabase


@pytest.fixture
async def filled_database(database, functions, embeddings):
    for function in functions:
        await database.add_function(
            function['name'],
            function['cfg'],
            embeddings[function['name']],
            binary_name=function['binary_name'],
            binary_sha256=function['binary_sha256'],
        )

    yield database


async def test_add_duplicate(database, functions, embeddings):
    function = functions[0]
    embedding = embeddings[function['name']]

    function_id = await database.add_function(
        function['name'],
        function['cfg'],
        embedding,
        binary_name=function['binary_name'],
        binary_sha256=function['binary_sha256'],
    )

    assert (
        await database.add_function(
            function['name'],
            function['cfg'],
            embedding,
            binary_name=function['binary_name'],
            binary_sha256=function['binary_sha256'],
        )
        == function_id
    )


async def test_add_binary_fields_optional(database, functions, embeddings):
    function = functions[-1]
    embedding = embeddings[function['name']]

    await database.add_function(function['name'], function['cfg'], embedding)
    assert await database.search_function(embedding) == [
        {
            'function': function['name'],
            'similarity': pytest.approx(1.0),
            'binary_name': None,
            'binary_sha256': None,
        }
    ]


async def test_add_user_id(filled_database, functions, embeddings):
    function = functions[-1]
    embedding = embeddings[function['name']]
    await filled_database.add_function(function['name'], function['cfg'], embedding, user_id='nobody@asmembedder.local')

    match filled_database:
        # Database interface exposes no raw query function, but we know the implementations
        case SQLiteDatabase():
            users = {row[0] for row in filled_database.connection.execute("""SELECT user_id FROM labels""")}
        case PostgreSQLDatabase():
            users = {row[0] for row in await filled_database.connection.fetch("""SELECT user_id FROM labels""")}
        case _:
            pytest.fail('unknown type of database implementation')

    assert users == {'nobody@asmembedder.local', None}


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
    query_embedding = query_embedding / numpy.linalg.norm(query_embedding)
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
        _init['name'],
        _init['cfg'],
        embeddings[_init['name']],
        binary_name='another_binary',
        binary_sha256='1234abcd' * 16,
    )

    results = await filled_database.search_function(embeddings[_init['name']], top_n=2)

    assert len(results) == 2
    assert '_init' == results[0]['function'] == results[1]['function']
    assert results[0]['binary_name'] != results[1]['binary_name']
