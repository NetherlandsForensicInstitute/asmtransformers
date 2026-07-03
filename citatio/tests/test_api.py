import os

import asyncpg
import pytest
from fastapi.testclient import TestClient

from citatio.api import app


pytestmark = pytest.mark.postgresql


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
