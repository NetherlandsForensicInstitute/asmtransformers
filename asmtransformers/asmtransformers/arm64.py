import re
from collections.abc import Iterator

from networkx import DiGraph

from asmtransformers.operands import is_offset


CONDITION_CODES = (
    # equal / not equal
    'eq',
    'ne',
    # carry & unsigned
    'cs',
    'hs',
    'cc',
    'lo',
    # negative / positive
    'mi',
    'pl',
    # overflow
    'vs',
    'vc',
    # unsigned higher / lower
    'hi',
    'ls',
    # greater & lesser (+ equal)
    'gt',
    'ge',
    'lt',
    'le',
    # always (b.al == b?)
    'al',
)


BRANCH_INSTRUCTIONS = (
    # direct branch
    'b',
    'br',
    # branch + link
    'bl',
    'blr',
    # branch on compare to zero
    'cbz',
    'cbnz',
    # return
    'ret',
    # test zero bit
    'tbz',
    'tbnz',
)
# conditional branches
BRANCH_INSTRUCTIONS += tuple(f'b.{cc}' for cc in CONDITION_CODES)


# a separator between operands; commas or whitespaces or a combination of both
OPERAND_SEPARATOR = re.compile(r'[,\s]+')


class Preprocessor:
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

    def preprocess(self, function_blocks: dict[int, list[str]] | DiGraph) -> list[str]:
        # collect token offsets for each basic block being processed as {block id → token offset}
        block_offsets = {}
        # collect token offsets for tokens that need to be patched to jump tokens {token offset → block id}
        jump_offsets = {}
        # collect all the tokens from the instructions
        # start with an empty token list unless we've been handed a prefix (e.g. 'CLS'), making sure that the token
        # offsets we're collecting line up with the token offsets where those blocks actually start (see below)
        tokens = list(self.prefix_tokens)

        if isinstance(function_blocks, DiGraph):
            # function_blocks is still in graph form, assume it's in jTrans form
            # all we need is a mapping of a block's id or offset to its assembly code
            function_blocks = {
                block_id: block['asm']
                for block_id, block
                # sort the graph by block id to force deterministic results
                in sorted(function_blocks.nodes.items())
            }

        for block_id, block in function_blocks.items():
            # log the 'next' token offset as the start of the block that will be processed next
            block_offsets[block_id] = len(tokens)
            for instruction in block:
                # parse the line of assembly into an instruction and its operands
                instruction, operands = parse_instruction(instruction)

                tokens.append(instruction)
                for operand in operands:
                    if instruction in BRANCH_INSTRUCTIONS and is_offset(operand):
                        # an operand to a branching instruction that is formatted as a hexadecimal number
                        # this is interpreter as the offset to a basic block, and this tracked as such in path_offsets
                        jump_target = int(operand[2:], base=16)
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
            case '[':
                # a dereference from the expression between brackets, provide as separate tokens surrounded by the
                # reference brackets
                # NB: this assumes there will be no nesting of bracketry, as that would slice an incorrect substring
                end = operands.index(']', offset)
                yield '['
                yield from parse_operands(operands[offset + 1 : end].lower())
                yield ']'
                # next offset is after the reference
                offset = end + 1
            case '!':
                # some referenced operands are postfixed with an exclamation mark
                # this is treated as an operand in itself
                yield '!'
                offset += 1
            case '{':
                # a set of (potentially partial) registers to which the instructions is applied, provide as separate
                # tokens surrounded by the set brackets
                # NB: this assumes there will be no nesting of bracketry, as that would slice an incorrect substring
                end = operands.index('}', offset)
                yield '{'
                yield from parse_operands(operands[offset + 1 : end].lower())
                yield '}'
                # next offset is after the set
                offset = end + 1
            # treat both spaces and commas as separators (skip these tokens)
            case ' ':
                offset += 1
            case ',':
                offset += 1
            case _:
                # any other case is 'just an operand'
                end = end.start() if (end := OPERAND_SEPARATOR.search(operands, offset)) else len(operands)
                yield operands[offset:end].lower()
                # next offset is either a separator (which will get ignored in the next iteration) or past the end of
                # the string (causing tokenization to end)
                offset = end
