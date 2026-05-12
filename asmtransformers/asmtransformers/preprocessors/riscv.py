import re
from collections.abc import Iterable

from asmtransformers.preprocessors import ASMPreprocessor


# useful info:
# https://projectf.io/posts/riscv-cheat-sheet/

# branch instructions will be treated differently, as we need to convert their addresses into jump address tokens
BRANCH_INSTRUCTIONS = {
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
    # call
    'call',
    # compressed instructions
    # jumps
    'c.j',
    'c.jal',
    # branches
    'c.beqz',
    'c.bnez',
    # register relative jumps are removed from this list as we can't resolve them with jump addresses
    # 'jalr',
    # 'c.jalr',
    # 'c.jr',
    # 'ret',
}


# a separator between operands; commas or whitespaces or a combination of both
OPERAND_SEPARATOR = re.compile(r'[,\s()]+')


class RISCVPreprocessor(ASMPreprocessor):
    branch_instructions = BRANCH_INSTRUCTIONS

    def parse_operands(self, operands: str) -> Iterable[str]:
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
                    yield from self.parse_operands(operands[offset + 1 : end].lower())
                    yield ')'
                    # next offset is after the reference
                    offset = end + 1
                case ' ' | ',':
                    # treat both spaces and commas as separators (skip these tokens)
                    offset += 1
                case _:
                    # any other case is 'just an operand'
                    # find the starting index of the separator; marking the end of the current operand
                    end = sep.start() if (sep := OPERAND_SEPARATOR.search(operands, offset)) else len(operands)
                    yield operands[offset:end].lower()
                    # next offset is either a separator (which will get ignored in the next iteration) or past the end
                    # of the string (causing tokenization to end)
                    offset = end
