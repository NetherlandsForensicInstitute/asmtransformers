import importlib.resources
import json

import pytest
from networkx import DiGraph

from asmtransformers import arm64, riscv
from asmtransformers.models.asmbert import ARM64Tokenizer


@pytest.fixture
def tokenizer():
    return riscv.RISCVPreprocessor()


def test_parse_no_operands():
    assert riscv.parse_instruction('ret') == ('ret', ())
    assert riscv.parse_instruction('c.nop') == ('c.nop', ())

# todo: find actual examples
# j is the only instruction that takes one operand, I couldn't find examples of actual operands that were not
# 'variablified'
def test_parse_single_operand():
    assert riscv.parse_instruction('j x0') == ('j', ('x0',))
    assert riscv.parse_instruction('c.j 0x123456') == ('c.j', ('0x123456',))


def test_parse_multiple_operands():
    assert arm64.parse_instruction('c.addi4spn s0,sp,0x30') == ('c.addi4spn', ('s0', 'sp', '0x30'))
    assert arm64.parse_instruction('ld a5,-0x28') == ('ld', ('a5', '-0x28'))
