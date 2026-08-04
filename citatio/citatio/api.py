from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated

import confidence
from asmtransformers import Architecture
from asmtransformers.models.embedder import ASMEmbedder
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.params import Body
from fastapi_oidc import IDToken, get_auth

from citatio.db import Database, PostgreSQLDatabase, SQLiteDatabase
from citatio.models import ControlFlowGraph


SUPPORTED_AUTH_MODES = frozenset({'anonymous', 'client_supplied', 'oidc'})
DEFAULT_MODEL = 'NetherlandsForensicInstitute/ARM64BERT-embedding'


def resolve_auth(**auth):
    # collect enabled auth modes from keywords; a truthy value enables the mode
    allowed = {mode for mode, enabled in auth.items() if enabled}
    if not allowed:
        raise ValueError('no enabled auth modes')
    if unsupported := (allowed - SUPPORTED_AUTH_MODES):
        raise ValueError(f'unsupported auth mode: {", ".join(unsupported)}')

    match auth:
        case {'oidc': oidc} if oidc:
            # create an OIDC Authorization header → IDToken function from the configured authentication settings
            return allowed, get_auth(**oidc)
        case _:
            # auth either not set or explicitly turned off, raise exception on presence of Authorization header
            def _oidc_unavailable(*args, **kwargs):
                raise HTTPException(503, 'Authentication unavailable')

            return allowed, _oidc_unavailable


def load_model(**model):
    match model:
        case {'hf': name_or_path} | {'path': name_or_path} if name_or_path:
            # from_pretrained takes either a name or a path, allow it to be specified either way, even though we can't
            # supply it explicitly as a name or a path
            return ASMEmbedder.from_pretrained(name_or_path)
        case _:
            raise ValueError('missing model configuration')


async def connect_database(**database) -> Database:
    match database:
        case {'engine': 'postgresql', 'postgresql': connect}:
            # database settings for postgresql, use PostgreSQLDatabase
            return await PostgreSQLDatabase.connect(**connect)
        case {'engine': 'sqlite', 'sqlite': name}:
            # explicit sqlite name to connect to, use SQLiteDatabase
            return await SQLiteDatabase.connect(name)
        case _:
            raise ValueError('missing database configuration')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load defaults from citatio module
    defaults = confidence.loads(resources.read_text('citatio', 'defaults.toml'), format=confidence.TOML)
    # combine defaults with user-supplied configuration
    app.state.config = config = defaults | confidence.load_name('citatio', format=confidence.TOML)

    app.state.model = load_model(**config.model)

    app.state.identification_modes, app.state.authenticate_user = resolve_auth(**config.auth)

    async with await connect_database(**config.database) as database:
        app.state.database = database
        yield


async def authenticated_user(request: Request) -> IDToken | None:
    if auth := request.headers.get('Authorization'):
        # authorization header available, let auth create a token from it
        # NB: this requires the token to be valid if it's supplied
        return request.app.state.authenticate_user(auth)
    else:
        # anonymous request, no token (this is explicitly allowed)
        return None


def identify_user(
    request: Request,
    user_id: Annotated[str | None, Body()] = None,
    id_token: Annotated[IDToken | None, Depends(authenticated_user)] = None,
):
    allowed = request.app.state.identification_modes

    match user_id, id_token:
        case str(), None if 'client_supplied' in allowed:
            return user_id
        case None, IDToken() if 'oidc' in allowed:
            return id_token.sub
        case None, None if 'anonymous' in allowed:
            return None

    raise HTTPException(401, {'error': 'no single identifiable user in allowed modes', 'allowed': sorted(allowed)})


app = FastAPI(lifespan=lifespan)


@app.get('/api/v1/auth/configuration')
async def auth_config(request: Request):
    # collect enabled authentication modes as booleans
    auth = {mode: mode in request.app.state.identification_modes for mode in SUPPORTED_AUTH_MODES}
    if auth['oidc']:
        # replace true with actionable config for oidc
        oidc = request.app.state.config.auth.oidc
        auth['oidc'] = {
            'client_id': oidc.client_id,
            'provider_uri': oidc.base_authorization_server_uri,
            'issuer': oidc.issuer,
        }

    return auth


@app.post('/api/v1/functions')
async def add_function(
    request: Request,
    label: Annotated[str, Body()],
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: Annotated[Architecture, Body()],
    binary_name: Annotated[str | None, Body()] = None,
    binary_sha256: Annotated[str | None, Body()] = None,
    user_id: Annotated[str | None, Depends(identify_user)] = None,
):
    embedding = request.app.state.model.encode(ControlFlowGraph.to_str(cfg), architecture=architecture)
    await request.app.state.database.add_function(
        label,
        architecture,
        cfg,
        embedding,
        user_id=user_id,
        binary_name=binary_name,
        binary_sha256=binary_sha256,
    )


@app.post('/api/v1/functions/search')
async def search_function(
    request: Request,
    cfg: Annotated[ControlFlowGraph, Body()],
    architecture: Annotated[Architecture, Body()],
    top_n: Annotated[int, Body()] = 25,
):
    embedding = request.app.state.model.encode(ControlFlowGraph.to_str(cfg), architecture=architecture)
    return await request.app.state.database.search_functions(embedding, top_n)
