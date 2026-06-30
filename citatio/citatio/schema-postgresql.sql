CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY,
    -- UNIQUE constraint allows using an error trigger as a 'cfg is already known' signal
    cfg TEXT NOT NULL UNIQUE,
    embedding VECTOR(768)
);

CREATE INDEX IF NOT EXISTS ON functions USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    -- (binary_name, binary_sha256) to identify the source of this label
    binary_name TEXT NOT NULL,
    binary_sha256 TEXT NOT NULL,

    FOREIGN KEY (function_id) REFERENCES functions(id)
);
