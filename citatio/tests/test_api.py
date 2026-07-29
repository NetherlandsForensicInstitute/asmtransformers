import pytest
from fastapi.testclient import TestClient

from citatio.api import app


@pytest.fixture
async def client(monkeypatch, database_env):
    # configure a test model to be loaded, avoid calling ASMEmbedder.from_pretrained
    monkeypatch.setenv('CITATIO_MODEL', '":test:"')
    # add anonymous and client_supplied authentication modes during test
    monkeypatch.setenv('CITATIO_AUTH_ANONYMOUS', 'true')
    monkeypatch.setenv('CITATIO_AUTH_CLIENT__SUPPLIED', 'true')

    with TestClient(app) as client:
        yield client


def test_get_auth_config(client):
    response = client.get('/api/v1/auth')
    assert response.status_code == 200
    assert response.json() == {'anonymous': True, 'client_supplied': True, 'oidc': False}


def test_get_auth_config_oidc(monkeypatch, database_env):
    monkeypatch.setenv('CITATIO_MODEL', '":test:"')
    # disable the silly auth
    monkeypatch.setenv('CITATIO_AUTH_ANONYMOUS', 'false')
    monkeypatch.setenv('CITATIO_AUTH_CLIENT__SUPPLIED', 'false')
    # configure an OIDC provider at example.com
    monkeypatch.setenv('CITATIO_AUTH_OIDC_CLIENT__ID', 'Cl1eNt-1D')
    monkeypatch.setenv('CITATIO_AUTH_OIDC_BASE__AUTHORIZATION__SERVER__URI', 'https://example.com/auth')
    monkeypatch.setenv('CITATIO_AUTH_OIDC_ISSUER', 'example.com')
    # cache ttl is required, but omitted from the response content
    monkeypatch.setenv('CITATIO_AUTH_OIDC_SIGNATURE__CACHE__TTL', '3600')

    with TestClient(app) as client:
        response = client.get('/api/v1/auth')
        assert response.status_code == 200
        assert response.json() == {
            'anonymous': False,
            'client_supplied': False,
            'oidc': {
                'client_id': 'Cl1eNt-1D',
                'provider_uri': 'https://example.com/auth',
                'issuer': 'example.com',
            },
        }


def test_no_auth(client, functions):
    response = client.post('/api/v1/functions', headers={'Authorization': 'Bearer R1ghtT0B34r4RMs'}, json=functions[0])
    assert response.is_server_error
    assert response.status_code == 503


def test_add_function(client, functions):
    response = client.post('/api/v1/functions', json=functions[0])
    assert response.status_code == 200


def test_add_function_supplied_user_id(client, functions):
    function = {**functions[0], 'user_id': 'GreatDane'}
    response = client.post('/api/v1/functions', json=function)
    assert response.status_code == 200


def test_add_function_anonymous_not_allowed(monkeypatch, client, functions):
    # disallow anonymous addition
    monkeypatch.setattr(app.state, 'identification_modes', {'client_supplied'})
    response = client.post('/api/v1/functions', json=functions[0])
    assert response.status_code == 401
    response = client.post('/api/v1/functions', json={**functions[1], 'user_id': 'MiniDane'})
    assert response.status_code == 200


def test_search_known(client, functions):
    for function in functions:
        client.post('/api/v1/functions', json=function)

    for function in functions:
        results = client.post('/api/v1/functions/search', json={'cfg': function['cfg'], 'top_n': 2}).json()
        assert len(results) == 2
        assert results[0]['similarity'] == pytest.approx(1.0)


def test_search_unknown(client, functions):
    for function in functions[1:]:
        client.post('/api/v1/functions', json=function)

    results = client.post('/api/v1/functions/search', json={'cfg': functions[0]['cfg']}).json()
    assert len(results) == 3
    for result in results:
        # nothing matches exactly, nothing should come back < 0.0
        assert 0.0 < result['similarity'] < 1.0
