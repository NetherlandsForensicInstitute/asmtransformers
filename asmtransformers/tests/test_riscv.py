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


# def test_offset_prefix_tokens(tokenizer):
#     graph = DiGraph()
#     graph.add_node(0x12, asm=['b 0x34'])
#     graph.add_node(0x34, asm=['b 0x12'])
#
#     tokens1 = tokenizer.preprocess(graph)
#     tokenizer.prefix_tokens = ('[CLS]', '[PAD]')
#     tokens2 = tokenizer.preprocess(graph)
#
#     assert tokens1 != tokens2
#     assert tokens2[:2] == ['[CLS]', '[PAD]']
#     # code is the same, jumps should have shifted due to prefixed tokens
#     assert (tokens1[1], tokens1[3]) == ('JUMP_ADDR_2', 'JUMP_ADDR_0')
#     assert (tokens2[3], tokens2[5]) == ('JUMP_ADDR_4', 'JUMP_ADDR_2')
#
#
# def test_format_operand():
#     class ObfuscatingTokenizer(arm64.ARM64Preprocessor):
#         def format_operand(self, operand):
#             if arm64.is_offset(operand):
#                 return 'OBFUSCATED'
#             else:
#                 return operand
#
#     graph = DiGraph()
#     graph.add_node(0x12, asm=['add x0,x0,0x78', 'b 0x78'])
#     graph.add_node(0x78, asm=['sub w0,w0,0x78', 'ret'])
#
#     tokens = ObfuscatingTokenizer().preprocess(graph)
#
#     # expect a jump token towards the second basic block, but expect the other two occurrences of 0x78 to have been
#     # obfuscated by obfuscate_offset
#     assert 'JUMP_ADDR_6' in tokens
#     assert '0x78' not in tokens
#     assert tokens.count('OBFUSCATED') == 2
#
#
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
