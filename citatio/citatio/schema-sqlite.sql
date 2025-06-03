/*
   Sample schema for use with SQLite + SQLite-vec
   Designed to be used for the following use cases:

   - Adding a function based on its CFG, expecting the function as (CFG, embedding, label, binary + digest, label)
   - Searching labels for a function based on a function's embedding
 */

CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY,
    -- UNIQUE constraint allows using an error trigger as a 'cfg is already known' signal
    cfg TEXT NOT NULL UNIQUE
);

CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0 (
    -- embeddings table has to be per-model
    -- PRIMARY KEY as opposed to FOREIGN KEY
    --   (which are not allowed for virtual tables, requiring manual management of this)
    function_id INTEGER PRIMARY KEY,
    -- embedding size has to be hard-coded (see above, expecting a table per model)
    embedding FLOAT[768]
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    -- (binary_name, binary_sha256) to identify the source of this label
    binary_name TEXT NOT NULL,
    binary_sha256 TEXT NOT NULL,

    FOREIGN KEY (function_id) REFERENCES functions(id)
);
