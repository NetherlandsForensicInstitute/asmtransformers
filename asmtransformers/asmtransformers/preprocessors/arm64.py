import re
from collections.abc import Iterable

from asmtransformers.preprocessors import ASMPreprocessor


CONDITION_CODES = {
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
}


BRANCH_INSTRUCTIONS = {
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
}
# conditional branches
BRANCH_INSTRUCTIONS |= {f'b.{cc}' for cc in CONDITION_CODES}


# a separator between operands; commas or whitespaces or a combination of both
OPERAND_SEPARATOR = re.compile(r'[,\s]+')


class ARM64Preprocessor(ASMPreprocessor):
    branch_instructions = BRANCH_INSTRUCTIONS

    def parse_operands(self, operands: str) -> Iterable[str]:
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
                    yield from self.parse_operands(operands[offset + 1 : end].lower())
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
                    yield from self.parse_operands(operands[offset + 1 : end].lower())
                    yield '}'
                    # next offset is after the set
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
