import sqlite3
import struct
from itertools import pairwise

import numpy as np
import pytest
import sqlite_vec

from citatio.db import SQLiteDatabase


@pytest.fixture
def mem_db():
    # wrap test database in a with statement to make sure the connection is closed after the test
    # (closing the in-memory database makes sure a re-connection for the next test results in an empty database)
    with sqlite3.connect(':memory:') as db:
        yield db


@pytest.fixture
def vec_mem_db(mem_db):
    mem_db.enable_load_extension(True)
    sqlite_vec.load(mem_db)
    mem_db.enable_load_extension(False)

    mem_db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0 (
        embedding FLOAT[2] distance_metric = cosine
    )
    """)

    for vector in ([-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]):
        mem_db.execute(
            'INSERT INTO embeddings (embedding) VALUES (?)',
            (sqlite_vec.serialize_float32(vector),)
        )

    yield mem_db


@pytest.fixture
def database(mem_db):
    yield SQLiteDatabase(mem_db)


@pytest.fixture
def filled_database(database, functions, embeddings):
    for function in functions:
        database.add_function(
            function['name'],
            function['cfg'],
            embeddings[function['name']],
            function['binary_name'],
            function['binary_sha256'],
        )

    yield database


def test_vec_distance(vec_mem_db):
    (a,), (b,), (c,) = list(vec_mem_db.execute(
        'SELECT distance FROM embeddings WHERE embedding MATCH ? AND K = 3 ORDER BY distance',
        (sqlite_vec.serialize_float32([-1.0, 0.0]),)
    ))
    # results ordered by distance, not insertion order
    assert (a, b, c) == pytest.approx((0.0, 1.0, 2.0))


def test_vec_scaled_distance(vec_mem_db):
    (da, ea), (db, eb), (dc, ec) = list(vec_mem_db.execute(
        """
        SELECT (2 - distance) / 2 AS similarity, embedding 
        FROM embeddings WHERE embedding MATCH ? AND K = 3 ORDER BY distance
        """,
        (sqlite_vec.serialize_float32([1.0, 0.0]),)
    ))
    # results ordered by distance, not insertion order
    assert (da, db, dc) == pytest.approx((1.0, 0.5, 0.0))
    assert struct.unpack('2f', ea) == (1.0, 0.0)
    assert struct.unpack('2f', eb) == (0.0, -1.0)
    assert struct.unpack('2f', ec) == (-1.0, 0.0)


def test_add_duplicate(database, functions, embeddings):
    function = functions[0]
    embedding = embeddings[function['name']]

    function_id = database.add_function(
        function['name'],
        function['cfg'],
        embedding,
        function['binary_name'],
        function['binary_sha256'],
    )

    assert database.add_function(
        function['name'],
        function['cfg'],
        embedding,
        function['binary_name'],
        function['binary_sha256'],
    ) == function_id


def test_search_identical(filled_database, embeddings):
    results = filled_database.search_function(embeddings['init_have_lse_atomics'])

    assert results[0]['function'] == 'init_have_lse_atomics'
    # similarity for that function should be 1.0
    assert results[0]['similarity'] == pytest.approx(1.0)
    # similarities should be declining throughout the results
    assert all(
        a['similarity'] >= b['similarity']
        for a, b in pairwise(results)
    )


def test_search_similar(filled_database, embeddings):
    jitter = np.random.rand(768).astype(dtype=np.float32) / 1_000.0
    query_embedding = embeddings['thunk_1234'] + jitter
    # normally, embeddings should have been normalized by the model, make sure to manually normalize the query here
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    results = filled_database.search_function(query_embedding)

    assert results[0]['function'] == 'thunk_1234'
    assert 0.9 < results[0]['similarity'] < 1.0


def test_search_top_1(filled_database, embeddings):
    results = filled_database.search_function(embeddings['init_have_lse_atomics'], top_n=1)

    assert len(results) == 1


def test_search_duplicate_label(filled_database, functions, embeddings):
    _init = next(function for function in functions if function['name'] == '_init')
    # add the same function with the same label as if it were from a different binary
    filled_database.add_function(_init['name'], _init['cfg'], embeddings[_init['name']], 'another_binary', '1234abcd' * 16)

    results = filled_database.search_function(embeddings[_init['name']], top_n=2)

    assert len(results) == 2
    assert '_init' == results[0]['function'] == results[1]['function']
    assert results[0]['binary_name'] != results[1]['binary_name']
