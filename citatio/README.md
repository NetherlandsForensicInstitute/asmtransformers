Citatio 📜
==========

This package provides a REST API to the [`asmtransformers`](../asmtransformers) and its 
[`ARM64BERT-embedding`](https://huggingface.co/NetherlandsForensicInstitute/ARM64BERT-embedding/) model, 
bridging Ghidra to a search index to find similar functions in ARM64 binaries.
See [`sententia`](../sententia) for the Ghidra plugin that uses this REST API to both add functions to the search index 
and search for possible function names / labels based on vector similarity.

Configuration and runtime
-------------------------

The citatio REST API takes 2 configuration options:

- The model to be used for embedding (currently, only `NetherlandsForensicInstitute/ARM64BERT-embedding` is supported);
- The database to store both assembly and embeddings in (currently, only SQLite is implemented).

Both of these can be configured through environment variables:

- `CITATIO_MODEL`: a local path or huggingface model name (though again, currently on the `ARM64BERT-embedding` model is supported);
- `CITATIO_SQLITE_DATABASE`: either `:memory:` or a local path to a SQLite database (will be created if it doesn't currently exist).

The defaults for these values are `NetherlandsForensicInstitute/ARM64BERT-embedding` and `:memory:`, 
resulting in a functioning but non-persistent service.

> [!NOTE]  
> After observing concurrency issues with SQLite and `sqlite-vec`, the REST API is currently served fully serialized and is consequently fairly slow.
> We're expecting to be able to solve this by using PostgreSQL and `pgvector`.

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

Python 3.12 or newer with SQLite version 3.35.0 or newer.

Requirements
------------

Installing this project locally can be done using `pip`:

```
$ python3 -m pip install .
```

For further development, this project uses [PDM](https://pdm-project.org/en/latest/) and `pyproject.toml` to manage dependencies.
See [PDM's installation instructions](https://pdm-project.org/en/latest/#installation) to get started, 
and subsequently call `pdm install` from the project's directory to automatically create a new virtual environment with dependencies.
