import pytest
from fastapi.testclient import TestClient

from citatio.api import app


@pytest.fixture
async def client(monkeypatch, database_env):
    # configure a fake model to be loaded, avoid calling ASMEmbedder.from_pretrained
    monkeypatch.setenv('CITATIO_MODEL', '":fake:"')

    with TestClient(app) as client:
        yield client


def test_no_auth(client, functions):
    response = client.post('/api/v1/add', headers={'Authorization': 'Bearer R1ghtT0B34r4RMs'}, json=functions[0])
    assert response.is_server_error
    assert response.status_code == 503


def test_add_function(client, functions):
    response = client.post('/api/v1/add', json=functions[0])
    assert response.status_code == 200


def test_add_function_supplied_user_id(client, functions):
    function = {**functions[0], 'user_id': 'GreatDane'}
    response = client.post('/api/v1/add', json=function)
    assert response.status_code == 200


def test_add_function_anonymous_not_allowed(monkeypatch, client, functions):
    # disallow anonymous addition
    monkeypatch.setattr(app.state, 'identification_modes', {'supplied'})
    response = client.post('/api/v1/add', json=functions[0])
    assert response.status_code == 401


def test_search_known(client, functions):
    for function in functions:
        client.post('/api/v1/add', json=function)

    for function in functions:
        results = client.post('/api/v1/search', json={'cfg': function['cfg'], 'top_n': 2}).json()
        assert len(results) == 2
        assert results[0]['similarity'] == pytest.approx(1.0)


def test_search_unknown(client, functions):
    for function in functions[1:]:
        client.post('/api/v1/add', json=function)

    results = client.post('/api/v1/search', json={'cfg': functions[0]['cfg']}).json()
    assert len(results) == 3
    for result in results:
        # nothing matches exactly, nothing should come back < 0.0
        assert 0.0 < result['similarity'] < 1.0
