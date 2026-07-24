from contextlib import asynccontextmanager
from typing import Annotated

import confidence
from asmtransformers.models.embedder import ASMEmbedder
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.params import Body
from fastapi_oidc import IDToken, get_auth

from citatio.db import Database, PostgreSQLDatabase, SQLiteDatabase
from citatio.models import ControlFlowGraph, FakeEmbedder


DEFAULT_MODEL = 'NetherlandsForensicInstitute/ARM64BERT-embedding'


def resolve_auth(**auth):
    match auth:
        case {'oidc': oidc}:
            # create an OIDC Authorization header → IDToken function from the configured authentication settings
            return get_auth(**oidc)
        case _:
            # auth either not set or explicitly turned off, raise exception on presence of Authorization header
            def _oidc_unavailable(*args, **kwargs):
                raise HTTPException(503, 'Authentication unavailable')

            return _oidc_unavailable


def load_model(model):
    match model:
        # TODO: this is effectively test code, can we patch that at test time?
        case ':fake:':
            return FakeEmbedder()
        case _:
            return ASMEmbedder.from_pretrained(model)


async def connect_database(**connect) -> Database:
    match connect:
        case {'sqlite': name}:
            # explicit sqlite name to connect to, use SQLiteDatabase
            return await SQLiteDatabase.connect(name)
        case {} if connect:
            # database settings *not* mentioning sqlite, use PostgreSQLDatabase
            return await PostgreSQLDatabase.connect(**connect)
        case _:
            raise ValueError('missing database configuration')


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = confidence.load_name('citatio')

    app.state.model = load_model(config.model or DEFAULT_MODEL)
    app.state.authenticate_user = resolve_auth(**config.auth)
    # TODO: determine from config, apply stricter defaults
    app.state.identification_modes = {'anonymous', 'supplied', 'oidc'}

    async with await connect_database(**config.database) as database:
        app.state.database = database
        yield


async def authenticated_user(request: Request) -> IDToken | None:
    if auth := request.headers.get('Authorization'):
        # authorization header available, let auth create a token from it
        return request.app.state.authenticate_user(auth)
    else:
        # anonymous request, no token
        return None


def identify_user(
    request: Request,
    user_id: Annotated[str | None, Body()] = None,
    id_token: Annotated[IDToken | None, Depends(authenticated_user)] = None,
):
    allowed = request.app.state.identification_modes

    match user_id, id_token:
        case str(), None if 'supplied' in allowed:
            return user_id
        case None, IDToken() if 'oidc' in allowed:
            return id_token.sub
        case None, None if 'anonymous' in allowed:
            return None

    raise HTTPException(401, {'error': 'no single identifiable user in allowed modes', 'allowed': sorted(allowed)})


app = FastAPI(lifespan=lifespan)


@app.post('/api/v1/add')
async def add_function(
    request: Request,
    name: Annotated[str, Body()],
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: Annotated[str, Body()] = 'arm64',
    binary_name: Annotated[str | None, Body()] = None,
    binary_sha256: Annotated[str | None, Body()] = None,
    user_id: Annotated[str | None, Depends(identify_user)] = None,
):
    embedding = request.app.state.model.encode(str(cfg), architecture=architecture)
    await request.app.state.database.add_function(
        name,
        cfg,
        embedding,
        user_id=user_id,
        binary_name=binary_name,
        binary_sha256=binary_sha256,
    )


@app.post('/api/v1/search')
async def search_function(
    request: Request,
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: Annotated[str, Body()] = 'arm64',
    top_n: Annotated[int, Body()] = 25,
):
    embedding = request.app.state.model.encode(str(cfg), architecture=architecture)
    return await request.app.state.database.search_function(embedding, top_n)
