from contextlib import asynccontextmanager
from typing import Annotated

import confidence
from asmtransformers.models.embedder import ASMEmbedder
from fastapi import FastAPI, Request
from fastapi.params import Body

from citatio.db import PostgreSQLDatabase, SQLiteDatabase
from citatio.models import ControlFlowGraph


DEFAULT_MODEL = 'NetherlandsForensicInstitute/ARM64BERT-embedding'


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = confidence.load_name('citatio')

    match config:
        case {'database.sqlite': name}:
            # explicit sqlite name to connect to, use SQLiteDatabase
            database = await SQLiteDatabase.connect(name)
        case {'database': connect}:
            # database settings *not* mentioning sqlite, use PostgreSQLDatabase
            database = await PostgreSQLDatabase.connect(**connect)
        case _:
            raise ValueError(f'missing database configuration, available: {config}')

    app.state.model = ASMEmbedder.from_pretrained(config.model or DEFAULT_MODEL)
    async with database:
        app.state.database = database
        yield


app = FastAPI(lifespan=lifespan)


@app.post('/api/v1/add')
async def add_function(
    request: Request,
    name: Annotated[str, Body()],
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: str = 'arm64',
    binary_name: Annotated[str, Body()] = None,
    binary_sha256: Annotated[str, Body()] = None,
):
    embedding = request.app.state.model.encode(str(cfg), architecture=architecture)
    await request.app.state.database.add_function(
        name,
        cfg,
        embedding,
        user_id=None,  # TODO: use authenticated user id when available
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
