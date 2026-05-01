import pytest

from asmtransformers import riscv


@pytest.fixture
def tokenizer():
    return riscv.RISCVPreprocessor()


def test_parse_no_operands():
    assert riscv.parse_instruction('ret') == ('ret', ())
    # there are only 20 items that can have a c. extension, indicating the instruction should be saved in less memory
    # so we keep it attached to the instruction
    assert riscv.parse_instruction('c.nop') == ('c.nop', ())


def test_parse_single_operand():
    assert riscv.parse_instruction('j x032') == ('j', ('x032',))
    assert riscv.parse_instruction('c.j 0x02') == ('c.j', ('0x02',))
    # we want to separate (sp) like we do in ARM64, as it is attached to a number which would result in a large vocab
    assert riscv.parse_instruction('c.sdsp ra,0x05(sp)') == ('c.sdsp', ('ra', '0x05', '(', 'sp', ')')
)


def test_parse_multiple_operands():
    assert riscv.parse_instruction('c.addi4spn s0,sp,0x30') == ('c.addi4spn', ('s0', 'sp', '0x30'))
    assert riscv.parse_instruction('ld a5,-0x28') == ('ld', ('a5', '-0x28'))



def test_context_length_boundary():
    # use a content length that would include the first two instructions, having the third instruction fall outside the
    # scope
    tokens = riscv.RISCVPreprocessor(context_length=10).preprocess(
        {
            0x12: ['addi a5,s0,-0xe0', 'c.mv s8,a5', 'j 0x34'],
            0x34: ['lw a5,-0x114', 'c.andi a5,0x2', 'beq 0x56'],
            0x56: ['c.mv a3,a5', 'addi a5,s0,-0xb0', 'blt 0x12'],
        }
    )

    # both tokens should be present, the last block in the graph (jump to 0x56) is outside the context length
    assert tokens.index('JUMP_ADDR_EXCEEDED') - tokens.index('beq') == 1


def test_jump_to_unknown_block(tokenizer):
    tokens = tokenizer.preprocess({
            0x12: ['bgt 0x123', 'add x0 x0 #0x12'],
            0x34: ['beq 0x12', 'sub x0 x0 #0x12'],
        })

    assert tokens.index('UNK_JUMP_ADDR') - tokens.index('bgt') == 1


def test_offset_prefix_tokens(tokenizer):
    graph = {
        0x12: ['bgt 0x34'],
        0x34: ['beq 0x12'],
             }

    tokens1 = tokenizer.preprocess(graph)
    tokenizer.prefix_tokens = ('[CLS]', '[PAD]')
    tokens2 = tokenizer.preprocess(graph)

    assert tokens1 != tokens2
    assert tokens2[:2] == ['[CLS]', '[PAD]']
    # code is the same, jumps should have shifted due to prefixed tokens
    assert (tokens1[1], tokens1[3]) == ('JUMP_ADDR_2', 'JUMP_ADDR_0')
    assert (tokens2[3], tokens2[5]) == ('JUMP_ADDR_4', 'JUMP_ADDR_2')


def test_format_operand():
    class ObfuscatingTokenizer(riscv.RISCVPreprocessor):
        def format_operand(self, operand):
            if riscv.is_offset(operand):
                return 'OBFUSCATED'
            else:
                return operand

    # todo: w0 does not exist in riscv, make this actual riscv compliant
    graph = {
        0x12: ['add x0,x0,0x78', 'beq 0x78'],
        0x78: ['sub w0,w0,0x78', 'ret'],
    }

    tokens = ObfuscatingTokenizer().preprocess(graph)
    # expect a jump token towards the second basic block, but expect the other two occurrences of 0x78 to have been
    # obfuscated by obfuscate_offset
    assert 'JUMP_ADDR_6' in tokens
    assert '0x78' not in tokens
    assert tokens.count('OBFUSCATED') == 2

# todo: adjust to risc-v; requires a riscv subfolder under asmtransformers/models
# def test_arm64_tokenizer_masks_padding_tokens():
#     tokenizer_path = importlib.resources.files('asmtransformers.models').joinpath('arm64bert')
#     tokenizer = ARM64Tokenizer.from_pretrained(tokenizer_path)
#
#     encoded = tokenizer([json.dumps([[0, ['ret']]])])
#     input_ids = encoded['input_ids'][0]
#     attention_mask = encoded['attention_mask'][0]
#
#     assert input_ids.shape[0] == 512
#     assert attention_mask.shape[0] == 512
#     assert attention_mask.sum().item() == 1
#     assert (input_ids[1:] == tokenizer.pad_token_id).all()
#     assert (attention_mask[1:] == 0).all()
