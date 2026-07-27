Citatio 📜
==========

This package provides a REST API to the [`asmtransformers`](../asmtransformers) and its 
[`ARM64BERT-embedding`](https://huggingface.co/NetherlandsForensicInstitute/ARM64BERT-embedding/) model, 
bridging Ghidra to a search index to find similar functions in ARM64 binaries.
See [`sententia`](../sententia) for the Ghidra plugin that uses this REST API to both add functions to the search index 
and search for possible function names / labels based on vector similarity.

Configuration and runtime
-------------------------

The citatio REST API takes 3 configuration options:

- The model to be used for embedding (currently, only `NetherlandsForensicInstitute/ARM64BERT-embedding` is supported);
- The authentication modes to support, any of `anonymous`, `client_supplied` and `oidc`;
- The database to store both assembly and embeddings in, either SQLite+sqlitevec or PostgreSQL+pgvector;

Citatio uses [confidence](https://github.com/NetherlandsForensicInstitute/confidence/) to read configuration, so both strategically placed files and environment variables are supported:

- `CITATIO_MODEL`: a local path or huggingface model name (though again, currently only the `ARM64BERT-embedding` model is supported);
- `CITATIO_AUTH_ANONYMOUS` (allowing operation without identifying a user) and 
  `CITATIO_AUTH_CLIENT__SUPPLIED` (enabling a client to supply a user identity in a request body) can be set to `true` to enable them, 
  OIDC configuration requires at least four values, see below.
- when using SQLite: `CITATIO_DATABASE_SQLITE`: either `:memory:` or a local path to a SQLite database (will be created if it doesn't currently exist).
- when using PostgreSQL: either `CITATIO_DATABASE_HOST`, `..._PORT`, `..._USER`, `..._PASSWORD`, `..._DATABASE` to connect to the database in question,
  or `CITATIO_DATABASE_DSN` with the full connection url to connect to that same database.

The authentication and database configuration is required, the default model to be loaded is `NetherlandsForensicInstitute/ARM64BERT-embedding`.

OIDC configuration is delegated to [FastAPI-OIDC](https://github.com/HarryMWinters/fastapi-oidc#configuration), which takes at least the following:

- `client_id`: the client id for this service, as configured in the OIDC provider;
- `base_authorization_server_uri`: the OIDC provider's base uri (the part before `.well-known/oidc-configuration/`);
- `issuer`: a single or multiple token issuer identifier(s);
- `signature_cache_ttl`: internal cache timeout, in seconds.

> [!NOTE]  
> After observing concurrency issues with SQLite and `sqlite-vec`, the REST API is currently served fully serialized when using SQLite and is consequently fairly slow.

Running the REST API service follows the default FastAPI command line setup, 
where the application is available from the `citatio` module:

```
$ fastapi dev citatio  # runs a development server
$ fastapi run citatio  # runs a production server
```

See [requirements](#requirements) for installation of citatio with its dependencies 
(including `fastapi` and [`asmtransformers`](../asmtransformers)).

Both the development and production servers will host a Swagger documentation for the available API,
though end users are encouraged to use the [ready-made Ghidra plugin (sententia)](../sententia).

Prerequisites
-------------

Python 3.13 or newer with either SQLite version 3.35.0 or newer, or PostgreSQL with the pgvector extension available.

Requirements
------------

Installing this project locally can be done using `pip`:

```
$ python3 -m pip install .
```

For further development, this project uses [PDM](https://pdm-project.org/en/latest/) and `pyproject.toml` to manage dependencies.
See [PDM's installation instructions](https://pdm-project.org/en/latest/#installation) to get started, 
and subsequently call `pdm install` from the project's directory to automatically create a new virtual environment with dependencies.
