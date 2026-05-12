import re
from collections.abc import Iterable, Iterator

from asmtransformers.preprocessors import ASMPreprocessor


# based mostly on: https://en.wikipedia.org/wiki/List_of_x86_instructions
BRANCH_INSTRUCTIONS = {
    # unconditional jump and call/return
    'jmp',
    'call',
    # equal / zero
    'je',
    'jz',
    # not equal / not zero
    'jne',
    'jnz',
    # signed less / less-or-equal
    'jl',
    'jnge',
    'jle',
    'jng',
    # signed greater / greater-or-equal
    'jg',
    'jnle',
    'jge',
    'jnl',
    # unsigned above / above-or-equal (carry clear)
    'ja',
    'jnbe',
    'jae',
    'jnb',
    # unsigned below / below-or-equal (carry set)
    'jb',
    'jnae',
    'jbe',
    'jna',
    # sign / overflow
    'js',
    'jns',
    'jo',
    'jno',
    # parity
    'jp',
    'jpe',
    'jnp',
    'jpo',
    # cx/ecx/rcx zero (short-range loop exits)
    'jcxz',
    'jecxz',
    'jrcxz',
    # loop instructions (branch if cx != 0)
    'loop',
    'loope',
    'loopne',
}


SIZE_QUALIFIERS = {
    'byte',
    'word',
    'dword',
    'qword',
    'tbyte',
    'xmmword',
    'ymmword',
    'zmmword',
}


# a separator between operands; commas, whitespaces, memory expressions or a combination of these
OPERAND_SEPARATOR = re.compile(r'[,\s\[\]]+')
# Matches one token that stops before commas, spaces, or memory-expression operators
_OPERAND_TOKEN = re.compile(r'[^\s,+\-*\[\]]+')


class X86Preprocessor(ASMPreprocessor):
    branch_instructions = BRANCH_INSTRUCTIONS

    def parse_operands(self, operands: str) -> Iterable[str]:
        offset = 0
        length = len(operands)

        while offset < length:
            match operands[offset]:
                case ' ' | ',':
                    # treat both spaces and commas as separators (skip these tokens)
                    offset += 1
                case '[':
                    # expression between square brackets is a memory expression
                    # slice this out of the operands string and process it separately
                    end = operands.index(']', offset)
                    yield '['
                    yield from self.parse_memory_expression(operands[offset + 1 : end].strip())
                    yield ']'
                    offset = end + 1
                case _:
                    # any other case is 'just an operand'
                    # find the starting index of the separator; marking the end of the current operand
                    end = sep.start() if (sep := OPERAND_SEPARATOR.search(operands, offset)) else len(operands)
                    token = operands[offset:end]

                    if token in SIZE_QUALIFIERS:
                        # consume the mandatory following "ptr" keyword and merge into one token
                        rest = operands[end:]
                        # NB: a size qualifier *should* be followed by "ptr"
                        assert rest.lstrip().startswith('ptr')
                        yield f'{token}_ptr'
                        # continue after "ptr"
                        offset = end + rest.index('ptr') + 3
                    else:
                        # strip optional segment-override suffix ":" from the token
                        # TODO: would it be informational to keep that suffix instead?
                        yield token.removesuffix(':')
                        offset = end


    def parse_memory_expression(self, expr: str) -> Iterator[str]:
        """
        We need this because subtracting a memory expression in ghidra is done by [rbp + -0x8].
        Split a memory address expression into tokens.
        treating negative displacements like "-0x8" as a single token rather than an operator followed by a number.
        """
        offset = 0
        length = len(expr)

        while offset < length:
            match expr[offset]:
                case ' ':
                    offset += 1
                case '+' | '*':
                    yield expr[offset]
                    offset += 1
                case '-':
                    # unary minus: attach to the number that follows when preceded by an operator or start
                    prev = expr[:offset].rstrip()
                    if not prev or prev[-1] in ('+', '-', '*'):
                        m = _OPERAND_TOKEN.match(expr, offset + 1)
                        if m:
                            yield f'-{m.group().lower()}'
                            offset = m.end()
                            continue
                    yield '-'
                    offset += 1
                case _:
                    m = _OPERAND_TOKEN.match(expr, offset)
                    if m:
                        yield m.group().lower()
                        offset = m.end()
                    else:
                        offset += 1
