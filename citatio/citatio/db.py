import sqlite3
from importlib import resources

import asyncpg
import numpy as np
import sqlite_vec
from pgvector.asyncpg import register_vector


class Database:
    async def add_function(
        self,
        name: str,
        cfg: dict[int, list[str]],
        embedding: np.array,
        *,
        user_id: str | None = None,
        binary_name: str | None = None,
        binary_sha256: bytes | None = None,
    ):
        pass

    async def search_function(self, embedding: np.array, top_n: int = 25):
        pass


class SQLiteDatabase(Database):
    @classmethod
    async def connect(cls, name):
        # make sure to pass check_same_thread=False, Python 3.11+ has thread-safe sqlite
        connection = sqlite3.connect(name, check_same_thread=False)
        return cls(connection)

    def __init__(self, connection):
        self.connection = connection

        with self.connection:
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)

            schema = resources.read_text('citatio', 'schema-sqlite.sql')
            self.connection.executescript(schema)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        self.connection = None

    async def _insert_or_get_function(self, cursor, cfg, embedding):
        # coerce cfg to str for database storage (keep the cfg type responsible for that (de)serialization)
        parameters = (str(cfg),)
        try:
            cursor.execute("""INSERT INTO functions (cfg) VALUES (?) RETURNING id""", parameters)
            function_id = cursor.fetchone()[0]
            cursor.execute(
                """INSERT INTO embeddings (function_id, embedding) VALUES (?, ?)""",
                (function_id, embedding),
            )
        except sqlite3.IntegrityError:
            # cfg already present, select its id
            cursor.execute("""SELECT id FROM functions WHERE cfg = ?""", parameters)
            function_id = cursor.fetchone()[0]

        return function_id

    async def add_function(self, name, cfg, embedding, *, user_id=None, binary_name=None, binary_sha256=None):
        with self.connection:
            cursor = self.connection.cursor()
            function_id = await self._insert_or_get_function(cursor, cfg, embedding)
            cursor.execute(
                """
                INSERT INTO
                    labels (function_id, label, user_id, binary_name, binary_sha256)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (function_id, name, user_id, binary_name, binary_sha256),
            )

            # return the function id for convenience
            return function_id

    async def search_function(self, embedding, top_n=25):
        cursor = self.connection.cursor()
        cursor.execute(
            """
                -- distance range is 0–2, translate it to similarity with range 0–1
                SELECT label, (2 - distance) / 2 AS similarity, binary_name, binary_sha256
                FROM labels
                    JOIN embeddings ON labels.function_id = embeddings.function_id
                -- use top N as the value for K, use closest N embeddings
                WHERE embeddings.embedding MATCH :embedding AND K = :n
                ORDER BY similarity DESC
                -- use top N again: limit number of results to N (in case multiple labels mapped to the same embedding)
                LIMIT :n
            """,
            {'embedding': embedding, 'n': top_n},
        )
        # Make the result into a list of dicts
        return [
            dict(
                zip(
                    ['function', 'similarity', 'binary_name', 'binary_sha256'],
                    result,
                    strict=True,
                )
            )
            for result in cursor
        ]


class PostgreSQLDatabase(Database):
    def __init__(self, connection):
        self.connection = connection

    @classmethod
    async def connect(cls, **kwargs):
        connection = await asyncpg.connect(**kwargs)
        await connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await register_vector(connection)
        await connection.execute(resources.read_text('citatio', 'schema-postgresql.sql'))
        return cls(connection)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connection.close()

    async def add_function(self, name, cfg, embedding, *, user_id=None, binary_name=None, binary_sha256=None):
        async with self.connection.transaction():
            function_id = await self.connection.fetchval(
                # use PostgreSQL's conflict resolution to issue an update-or-get
                # NB: the conflict resolution update is idempotent, but needed to make sure RETURNING id works
                """
                INSERT INTO functions (cfg, embedding) VALUES ($1, $2)
                ON CONFLICT (cfg) DO UPDATE SET cfg = EXCLUDED.cfg RETURNING id
                """,
                str(cfg),
                embedding,
            )
            await self.connection.execute(
                """
                INSERT INTO
                    labels (function_id, label, user_id, binary_name, binary_sha256)
                VALUES
                    ($1, $2, $3, $4, $5)
                """,
                function_id,
                name,
                user_id,
                binary_name,
                binary_sha256,
            )

        return function_id

    async def search_function(self, embedding, top_n=25):
        results = await self.connection.fetch(
            """
            SELECT label, (2 - (embedding <=> $1)) / 2 AS similarity, binary_name, binary_sha256
            FROM labels
                JOIN functions ON labels.function_id = functions.id 
            ORDER BY similarity DESC
            LIMIT $2
            """,
            embedding,
            top_n,
        )

        return [
            dict(
                zip(
                    ['function', 'similarity', 'binary_name', 'binary_sha256'],
                    result,
                    strict=True,
                )
            )
            for result in results
        ]
