import json

import pytest

from citatio.models import Block, ControlFlowGraph


def test_empty():
    cfg = ControlFlowGraph([])

    assert not cfg
    assert len(cfg) == 0
    assert len(cfg.blocks) == 0
    assert len(list(cfg)) == 0
    assert str(cfg) == ControlFlowGraph.to_str(cfg) == '[]'


def test_list_roundtrip():
    data = [[0, ['add x1,x1', 'ret']], [12, ['ret']], [34, ['b 0']]]

    cfg = ControlFlowGraph(data)
    assert len(cfg.blocks) == len(data)
    assert len(cfg) == 4  # 3 blocks with a total of 4 instructions
    assert list(cfg) == data
    assert str(cfg).replace(', ', ',') == ControlFlowGraph.to_str(cfg).replace(', ', ',')
    assert str(cfg).replace(', ', ',') == '[[0,["add x1,x1", "ret"]], [12, ["ret"]], [34, ["b 0"]]]'.replace(', ', ',')


def test_str_roundtrip():
    data = '[[0, ["add x1,x1", "ret"]], [12, ["ret"]], [34, ["b 0"]]]'

    cfg = ControlFlowGraph.from_str(data)
    assert len(cfg) == 4  # 3 blocks with a total of 4 instructions
    # remove spaces from both sides to 'enable' fair ==
    assert str(cfg).replace(', ', ',') == data.replace(', ', ',') == ControlFlowGraph.to_str(cfg).replace(', ', ',')
    assert ControlFlowGraph.to_str(data) == data
    with pytest.raises(TypeError):
        ControlFlowGraph.to_str(data.encode('utf-8'))


def test_back_and_forth():
    cfg = ControlFlowGraph(
        [
            Block(0, ['add x1,x1', 'ret']),
            Block(12, ['ret']),
            Block(34, ['b 0']),
        ]
    )

    assert cfg == ControlFlowGraph.from_str(json.dumps(list(cfg)))
