from functools import partial

from networkx import DiGraph
import pytest

from asmtransformers import arm64


@pytest.fixture
def tokenizer():
    return arm64.Preprocessor()


def test_parse_no_operands():
    assert arm64.parse_instruction('ret') == ('ret', ())


def test_parse_single_operand():
    assert arm64.parse_instruction('blr x0') == ('blr', ('x0',))
    assert arm64.parse_instruction('b 0x123456') == ('b', ('0x123456',))
    assert arm64.parse_instruction('b.ne [w0, #0x4]') == ('b.ne', ('[', 'w0', '#0x4', ']'))


def test_parse_multiple_operands():
    assert arm64.parse_instruction('ldp x19,x20,[sp, #0x10]') == ('ldp', ('x19', 'x20', '[', 'sp', '#0x10', ']'))
    assert arm64.parse_instruction('ldp x29,x30,[sp], #0x60') == ('ldp', ('x29', 'x30', '[', 'sp', ']', '#0x60'))
    assert arm64.parse_instruction('movk x1,#0x4024, LSL #16') == ('movk', ('x1', '#0x4024', 'lsl', '#16'))
    assert arm64.parse_instruction('stp x29,x30,[sp, #-0x60]!') == ('stp', ('x29', 'x30', '[', 'sp', '#-0x60', ']', '!'))
    assert arm64.parse_instruction('ldp x29,x30,[sp], #0x60') == ('ldp', ('x29', 'x30', '[', 'sp', ']', '#0x60'))


def test_tokenize_single_block(tokenizer):
    graph = DiGraph()
    graph.add_node(0, asm=['ld x0,#0x1234', 'add x0,x0,#0x1234', 'ret'])

    assert tokenizer.preprocess(graph) == [
        'ld', 'x0', '#0x1234',
        'add', 'x0', 'x0', '#0x1234',
        'ret',
    ]


def test_tokenize_branching_blocks(tokenizer):
    graph = DiGraph()
    # NB: nodes are in 'reverse order', tokenizer should (?) reorder these based on their node ids
    graph.add_node(42, asm=['add x2,x2,#0x290', 'b.eq x2,0x0'])  # branch to offset 0
    graph.add_node(0, asm=['sub x2,x2,#0x290', 'bl 0x2a'])  # branch to offset 42

    assert tokenizer.preprocess(graph) == [
        'sub', 'x2', 'x2', '#0x290', 'bl', 'JUMP_ADDR_6',
        'add', 'x2', 'x2', '#0x290', 'b.eq', 'x2', 'JUMP_ADDR_0',
    ]


def test_context_length_boundary():
    # use a content length that would include the first two instructions, having the third instruction fall outside the scope
    tokens = arm64.Preprocessor(context_length=10).preprocess({
        0x12: ['ldp x19,x20,[sp, #0x10]', 'b 0x34'],
        0x34: ['movk x1,#0x4024, LSL #16', 'b.eq 0x56'],
        0x56: ['ldp x29,x30,[sp], #0x60', 'b.le 0x12'],
    })

    # both tokens should be present, the last block in the graph (jump to 0x56) is outside the context length
    assert tokens.index('JUMP_ADDR_EXCEEDED') - tokens.index('b.eq') == 1


def test_jump_to_unknown_block(tokenizer):
    graph = DiGraph()
    graph.add_node(0x12, asm=['b.gt 0x123', 'add x0 x0 #0x12'])
    graph.add_node(0x34, asm=['b 0x12', 'sub x0 x0 #0x12'])

    tokens = tokenizer.preprocess(graph)

    assert tokens.index('UNK_JUMP_ADDR') - tokens.index('b.gt') == 1


def test_offset_prefix_tokens(tokenizer):
    graph = DiGraph()
    graph.add_node(0x12, asm=['b 0x34'])
    graph.add_node(0x34, asm=['b 0x12'])

    tokens1 = tokenizer.preprocess(graph)
    tokenizer.prefix_tokens = ('[CLS]', '[PAD]')
    tokens2 = tokenizer.preprocess(graph)

    assert tokens1 != tokens2
    assert tokens2[:2] == ['[CLS]', '[PAD]']
    # code is the same, jumps should have shifted due to prefixed tokens
    assert (tokens1[1], tokens1[3]) == ('JUMP_ADDR_2', 'JUMP_ADDR_0')
    assert (tokens2[3], tokens2[5]) == ('JUMP_ADDR_4', 'JUMP_ADDR_2')


def test_format_operand():
    class ObfuscatingTokenizer(arm64.Preprocessor):
        def format_operand(self, operand):
            if arm64.is_offset(operand):
                return 'OBFUSCATED'
            else:
                return operand

    graph = DiGraph()
    graph.add_node(0x12, asm=['add x0,x0,0x78', 'b 0x78'])
    graph.add_node(0x78, asm=['sub w0,w0,0x78', 'ret'])

    tokens = ObfuscatingTokenizer().preprocess(graph)

    # expect a jump token towards the second basic block, but expect the other two occurrences of 0x78 to have been
    # obfuscated by obfuscate_offset
    assert f'JUMP_ADDR_6' in tokens
    assert '0x78' not in tokens
    assert tokens.count('OBFUSCATED') == 2
