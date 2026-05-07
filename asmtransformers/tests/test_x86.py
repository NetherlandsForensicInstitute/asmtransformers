import pytest

from asmtransformers import x86
from asmtransformers.operands import is_offset


@pytest.fixture
def tokenizer():
    return x86.X86Preprocessor()


def test_parse_plain_operands():
    assert list(x86.parse_operands('rax')) == ['rax']
    assert list(x86.parse_operands('0x10')) == ['0x10']
    assert list(x86.parse_operands('rax, rbx')) == ['rax', 'rbx']
    assert list(x86.parse_operands('rax, 0x10')) == ['rax', '0x10']


def test_parse_memory_operand():
    assert list(x86.parse_operands('[rax]')) == ['[', 'rax', ']']
    assert list(x86.parse_operands('[rbp + -0x8]')) == ['[', 'rbp', '+', '-0x8', ']']


def test_parse_size_qualifier():
    assert list(x86.parse_operands('dword ptr [rbp + -0x8]')) == ['dword_ptr', '[', 'rbp', '+', '-0x8', ']']
    assert list(x86.parse_operands('xmmword ptr [rax]')) == ['xmmword_ptr', '[', 'rax', ']']


def test_parse_segment_override():
    assert list(x86.parse_operands('fs:[rax]')) == ['fs', '[', 'rax', ']']


def test_parse_complex_memory():
    assert list(x86.parse_operands('[rax + rcx*4 + 0x10]')) == [
        '[',
        'rax',
        '+',
        'rcx',
        '*',
        '4',
        '+',
        '0x10',
        ']',
    ]


def test_tokenize_single_block(tokenizer):
    
    graph={0: ['mov rax, 0x1234', 'add rax, 0x1234', 'ret']}

    assert tokenizer.preprocess(graph) == [
        'mov',
        'rax',
        '0x1234',
        'add',
        'rax',
        '0x1234',
        'ret',
    ]


def test_tokenize_branching_blocks(tokenizer):

    # NB: nodes are in 'reverse order', tokenizer should reorder these based on their node ids
    graph = {
        42: ['add rcx, 0x290', 'je 0x0'], 
        0: ['sub rcx, 0x290', 'jmp 0x2a']
    }  # branch to offset 0

    assert tokenizer.preprocess(graph) == [
        'sub',
        'rcx',
        '0x290',
        'jmp',
        'JUMP_ADDR_5',
        'add',
        'rcx',
        '0x290',
        'je',
        'JUMP_ADDR_0',
    ]


def test_context_length_boundary():
    tokens = x86.X86Preprocessor(context_length=10).preprocess(
        {
            0x12: ['mov rax, 0x1234', 'jmp 0x34'],
            0x34: ['add rcx, 0x290', 'je 0x56'],
            0x56: ['sub rdx, 0x10', 'jmp 0x12'],
        }
    )

    # je target (0x56) falls at block offset 10, which equals context_length — exceeded
    assert tokens.index('JUMP_ADDR_EXCEEDED') - tokens.index('je') == 1


def test_jump_to_unknown_block(tokenizer):
    graph = {
        0x12 : ['jg 0x999', 'add rax, 0x12'],
        0x34: ['jmp 0x12', 'sub rax, 0x12']
    }
    tokens= tokenizer.preprocess(graph)

    assert tokens.index('UNK_JUMP_ADDR') - tokens.index('jg') == 1


def test_offset_prefix_tokens(tokenizer):
    
    graph= {
        0x12: ['jmp 0x34'],
        0x34 :['jmp 0x12']
    }
    tokens1 = tokenizer.preprocess(graph)
    tokenizer.prefix_tokens = ('[CLS]', '[PAD]')
    tokens2 = tokenizer.preprocess(graph)

    assert tokens1 != tokens2
    assert tokens2[:2] == ['[CLS]', '[PAD]']
    # same code, jumps shift by the two prefix tokens
    assert (tokens1[1], tokens1[3]) == ('JUMP_ADDR_2', 'JUMP_ADDR_0')
    assert (tokens2[3], tokens2[5]) == ('JUMP_ADDR_4', 'JUMP_ADDR_2')


def test_format_operand():
    class ObfuscatingPreprocessor(x86.X86Preprocessor):
        def format_operand(self, operand):
            if is_offset(operand):
                return 'OBFUSCATED'
            else:
                return operand

    graph = {
        0x12: ['add rax, 0x78', 'jmp 0x78'],
        0x78: ['sub rbx, 0x78', 'ret']
    }
    tokens = ObfuscatingPreprocessor().preprocess(graph)

    # jmp 0x78 resolves to a JUMP_ADDR token; the other two 0x78 occurrences are obfuscated
    assert 'JUMP_ADDR_5' in tokens
    assert '0x78' not in tokens
    assert tokens.count('OBFUSCATED') == 2
