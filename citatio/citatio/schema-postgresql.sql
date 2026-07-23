CREATE TABLE IF NOT EXISTS functions (
    id SERIAL PRIMARY KEY,
    -- UNIQUE constraint allows using an error trigger as a 'cfg is already known' signal
    cfg TEXT NOT NULL UNIQUE,
    embedding VECTOR(768)
);

CREATE INDEX IF NOT EXISTS embeddings_cosine ON functions USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS labels (
    id SERIAL PRIMARY KEY,
    function_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    user_id TEXT DEFAULT NULL,
    -- (binary_name, binary_sha256) to identify the source of this label
    binary_name TEXT DEFAULT NULL,
    binary_sha256 TEXT DEFAULT NULL,

    FOREIGN KEY (function_id) REFERENCES functions(id),
    -- enforce unique function labels per user
    -- NB: user_id allows NULL, which is DISTINCT by default, enabling multiple (function, label) combinations for
    --     anonymous users
    UNIQUE (function_id, user_id)
);
