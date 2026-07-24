import json
from collections.abc import Sequence
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from pydantic import RootModel


class Block(NamedTuple):
    offset: int
    instructions: Sequence[str]

    def __len__(self):
        return len(self.instructions)

    def __eq__(self, other):
        # care only about the size and values, not tuple vs NamedTuple
        return tuple(self) == tuple(other)


class ControlFlowGraph(RootModel[list[Block]]):
    @classmethod
    def from_str(cls, value):
        return cls(json.loads(value))

    @property
    def blocks(self):
        # RootModel requires the use of the "root" attribute, proxy it to "blocks" for clarity
        return self.root

    def __len__(self):
        # define the length of a CFG as the number of instructions it contains
        return sum(len(block) for block in self.blocks)

    def __iter__(self):
        # __iter__ is invoked on a list-coercion, list(cfg) would result in a list of values being yielded here
        return iter(self.blocks)

    def __str__(self):
        # __str__ is invoked on str-coercion, str(cfg) would result in a JSON-encoded list of blocks
        return json.dumps(self.blocks)


class FakeEmbedder:
    def encode(self, cfg, **kwargs):
        v = np.frombuffer(
            # string together enough sha256 hashes to gather a deterministic uint8 array of length 768
            b''.join(sha256(bytes(f'{n}: {cfg}', encoding='utf-8')).digest() for n in range(768 // 32)),
            dtype=np.uint8,
        )
        # normalize the array like a real model would
        # NB: make sure to explicitly use float32, sqlite assumes 4-byte values
        return np.divide(v, np.linalg.norm(v), dtype=np.float32)
