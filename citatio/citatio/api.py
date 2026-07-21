from contextlib import asynccontextmanager
from typing import Annotated

import confidence
from asmtransformers.models.embedder import ASMEmbedder
from fastapi import Depends, FastAPI, Request
from fastapi.params import Body
from fastapi_oidc import IDToken

from citatio.db import Database, PostgreSQLDatabase, SQLiteDatabase
from citatio.models import ControlFlowGraph


DEFAULT_MODEL = 'NetherlandsForensicInstitute/ARM64BERT-embedding'


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

    app.state.model = ASMEmbedder.from_pretrained(config.model or DEFAULT_MODEL)

    async with await connect_database(**config.database) as database:
        app.state.database = database
        yield

    # FIXME: resolve and set app.state.authenticate_user (created through fastapi_oidc.get_auth())


async def authenticated_user(request: Request) -> IDToken | None:
    if auth := request.headers.get('Authorization'):
        # authorization header available, let auth create a token from it
        return request.app.state.authenticate_user(auth)
    else:
        # anonymous request, no token
        return None


app = FastAPI(lifespan=lifespan)


@app.post('/api/v1/add')
async def add_function(
    request: Request,
    name: Annotated[str, Body()],
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: str = 'arm64',
    binary_name: Annotated[str | None, Body()] = None,
    binary_sha256: Annotated[str | None, Body()] = None,
    id_token: IDToken | None = Depends(authenticated_user),  # noqa: B008 (not a function call, just the FastAPI way™)
):
    embedding = request.app.state.model.encode(str(cfg), architecture=architecture)
    await request.app.state.database.add_function(
        name,
        cfg,
        embedding,
        # user_id is optional, use subject identifier from token if available
        user_id=id_token.sub if id_token else None,
        binary_name=binary_name,
        binary_sha256=binary_sha256,
    )


@app.post('/api/v1/search')
async def search_function(
    request: Request,
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: str = 'arm64',
    top_n: Annotated[int, Body()] = 25,
):
    embedding = request.app.state.model.encode(str(cfg), architecture=architecture)
    return await request.app.state.database.search_function(embedding, top_n)
