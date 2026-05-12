import re
from collections.abc import Iterator

from asmtransformers.operands import is_offset


# useful info:
# https://projectf.io/posts/riscv-cheat-sheet/

# branch instructions will be treated differently, as we need to convert their addresses into jump address tokens
BRANCH_INSTRUCTIONS = (
    # branch (not) equal to zero
    'beq',
    'bne',
    'beqz',
    'bnez',
    # less than
    'blt',
    'bltu',
    'bltz',
    # greater than
    'bgt',
    'bgtu',
    'bgtz',
    # less or equal
    'ble',
    'bleu',
    'blez',
    # greater or equal
    'bge',
    'bgeu',
    'bgez',
    # jump
    'j',
    'jal',
    'jalr',
    # return
    'ret',
    'call',
    # compressed instructions
    # jumps
    'c.j',
    'c.jal',
    'c.jalr',
    'c.jr',
    # branches
    'c.beqz',
    'c.bnez',
)


# a separator between operands; commas or whitespaces or a combination of both
OPERAND_SEPARATOR = re.compile(r'[,\s\(\)]+')


class RISCVPreprocessor:
    def __init__(self, *, context_length=512, prefix_tokens=None, operand_formatters=None):
        self.context_length = context_length
        self.prefix_tokens = prefix_tokens or ()
        self.operand_formatters = operand_formatters or ()

    def format_jump(self, operand: str, target_index: int | None) -> str:
        if target_index is None:
            return 'UNK_JUMP_ADDR'
        elif target_index < self.context_length:
            return f'JUMP_ADDR_{target_index}'
        else:
            return 'JUMP_ADDR_EXCEEDED'

    def format_operand(self, operand: str) -> str | None:
        for formatter in self.operand_formatters:
            # provide the first non-falsy value returned by any formatter
            if replacement := formatter(operand):
                return replacement

    def preprocess(self, function_blocks: dict[int, list[str]]) -> list[str]:
        # collect token offsets for each basic block being processed as {block id → token offset}
        block_offsets = {}
        # collect token offsets for tokens that need to be patched to jump tokens {token offset → block id}
        jump_offsets = {}
        # collect all the tokens from the instructions
        # start with an empty token list unless we've been handed a prefix (e.g. 'CLS'), making sure that the token
        # offsets we're collecting line up with the token offsets where those blocks actually start (see below)
        tokens = list(self.prefix_tokens)

        function_blocks = dict(sorted(function_blocks.items()))

        for block_id, block in function_blocks.items():
            # log the 'next' token offset as the start of the block that will be processed next
            block_offsets[block_id] = len(tokens)
            for instruction in block:
                # parse the line of assembly into an instruction and its operands
                instruction, operands = parse_instruction(instruction)

                tokens.append(instruction)
                for operand in operands:
                    if instruction in BRANCH_INSTRUCTIONS and (offset := is_offset(operand)):
                        # can't slice at place 2 because negative hex values so therefore use value from is_offset regex
                        jump_target = int(offset.group('value'), base=16)
                        jump_offsets[len(tokens)] = jump_target
                    else:
                        # for anything but an address operand of a branching / jumping instruction, let format_operand
                        # reformat the operand if it was supplied (but fall back to the original if the formatter
                        # returns nothing)
                        operand = self.format_operand(operand) or operand

                    tokens.append(operand)

        # all tokens are now collected, patch the tokens at the offsets collected before
        for offset, jump_target in jump_offsets.items():
            # let format_jump come up with a token for a jump to the token offset associated with the basic block that
            # it jumps to
            # NB: it might be jumping to a block that we don't know, let format_jump deal with that by passing it None
            #     in that case
            token = tokens[offset]
            if replacement := self.format_jump(token, block_offsets.get(jump_target)):
                tokens[offset] = replacement

        return tokens


def parse_instruction(instruction: str) -> tuple[str, tuple[str, ...]]:
    match instruction.split(maxsplit=1):
        # instruction and a number of operands to be parsed
        case instruction, operands:
            return instruction, tuple(parse_operands(operands))
        # no operands to be parsed (but instruction will be a list here)
        case instruction:
            return instruction[0], ()


def parse_operands(operands: str) -> Iterator[str]:
    # move through the string of operands linearly, starting at offset 0
    offset = 0
    while offset < len(operands):
        match operands[offset]:
            case '(':
                # a dereference from the expression between brackets, provide as separate tokens surrounded by the
                # reference brackets
                # NB: this assumes there will be no nesting of bracketry, as that would slice an incorrect substring
                end = operands.index(')', offset)
                yield '('
                yield from parse_operands(operands[offset + 1 : end].lower())
                yield ')'
                # next offset is after the reference
                offset = end + 1
            case ' ' | ',':
                # treat both spaces and commas as separators (skip these tokens)
                offset += 1
            case _:
                # any other case is 'just an operand'
                end = end.start() if (end := OPERAND_SEPARATOR.search(operands, offset)) else len(operands)
                yield operands[offset:end].lower()
                # next offset is either a separator (which will get ignored in the next iteration) or past the end of
                # the string (causing tokenization to end)
                offset = end
