import json

import numpy as np

from citatio.models import Block, ControlFlowGraph, FakeEmbedder


def test_empty():
    cfg = ControlFlowGraph([])

    assert not cfg
    assert len(cfg) == 0
    assert len(cfg.blocks) == 0
    assert len(list(cfg)) == 0
    assert str(cfg) == '[]'


def test_list_roundtrip():
    data = [[0, ['add x1,x1', 'ret']], [12, ['ret']], [34, ['b 0']]]

    cfg = ControlFlowGraph(data)
    assert len(cfg.blocks) == len(data)
    assert len(cfg) == 4  # 3 blocks with a total of 4 instructions
    assert list(cfg) == data


def test_str_roundtrip():
    data = '[[0, ["add x1,x1", "ret"]], [12, ["ret"]], [34, ["b 0"]]]'

    cfg = ControlFlowGraph.from_str(data)
    assert len(cfg) == 4  # 3 blocks with a total of 4 instructions
    # remove spaces from both sides to 'enable' fair ==
    assert str(cfg).replace(' ', '') == data.replace(' ', '')


def test_back_and_forth():
    cfg = ControlFlowGraph(
        [
            Block(0, ['add x1,x1', 'ret']),
            Block(12, ['ret']),
            Block(34, ['b 0']),
        ]
    )

    assert cfg == ControlFlowGraph.from_str(json.dumps(list(cfg)))


def test_fake_embedder():
    data = '[[0, ["add x1,x1", "ret"]], [12, ["ret"]], [34, ["b 0"]]]'
    embedder = FakeEmbedder()

    assert len(embedder.encode(data)) == 768
    assert np.allclose(embedder.encode(data), embedder.encode(data, architecture='mips'))
    assert np.allclose(FakeEmbedder().encode(data), embedder.encode(data))
    assert not np.allclose(embedder.encode('other data'), embedder.encode(data))
