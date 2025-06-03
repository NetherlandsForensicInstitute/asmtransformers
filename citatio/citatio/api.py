import os
from contextlib import asynccontextmanager
from typing import Annotated

from asmtransformers.models.asmsentencebert import ASMSentenceTransformer
from fastapi import FastAPI, Request
from fastapi.params import Body

from citatio.db import SQLiteDatabase
from citatio.models import ControlFlowGraph


DEFAULT_MODEL = 'NetherlandsForensicInstitute/ARM64BERT-embedding'


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = os.environ.get('CITATIO_MODEL', DEFAULT_MODEL)
    database = os.environ.get('CITATIO_SQLITE_DATABASE', ':memory:')

    if database == ':memory:':
        # TODO: log warning that we're running in in-memory database
        pass

    app.state.model = ASMSentenceTransformer.from_pretrained(model)

    with SQLiteDatabase.from_name(database) as database:
        app.state.database = database

        yield


app = FastAPI(lifespan=lifespan)


@app.post('/api/v1/add')
async def add_function(  # NB: function body isn't actually async, forcing it to run blocking on the event loop
    request: Request,
    name: Annotated[str, Body()],
    cfg: Annotated[ControlFlowGraph, Body()],
    binary_name: Annotated[str, Body()] = None,
    binary_sha256: Annotated[str, Body()] = None,
):
    embedding = request.app.state.model.encode(str(cfg))
    request.app.state.database.add_function(name, cfg, embedding, binary_name, binary_sha256)


@app.post('/api/v1/search')
async def search_function(  # NB: function body isn't actually async, forcing it to run blocking on the event loop
    request: Request,
    cfg: Annotated[ControlFlowGraph, Body()],
    top_n: Annotated[int, Body()] = 25,
):
    embedding = request.app.state.model.encode(str(cfg))
    return request.app.state.database.search_function(embedding, top_n)
