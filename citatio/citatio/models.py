import json
from collections.abc import Sequence
from typing import NamedTuple

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
    def from_str(cls, value: str):
        return cls(json.loads(value))

    @staticmethod
    def to_str(value):
        match value:
            case str():
                return value
            case [*blocks]:  # noqa: F841 (named "blocks" for clarity)
                return json.dumps(value)
            case ControlFlowGraph():
                return str(value)
            case _:
                raise TypeError

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
