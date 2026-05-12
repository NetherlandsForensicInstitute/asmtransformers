import re
from collections.abc import Iterable, Iterator

from asmtransformers.preprocessors import ASMPreprocessor


# based mostly on: https://en.wikipedia.org/wiki/List_of_x86_instructions
BRANCH_INSTRUCTIONS = (
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
)


SIZE_QUALIFIERS = (
    'byte',
    'word',
    'dword',
    'qword',
    'tbyte',
    'xmmword',
    'ymmword',
    'zmmword',
)

_SIZE_QUALIFIER_SET = frozenset(SIZE_QUALIFIERS)

# Matches one token that stops before commas, spaces, or memory-expression operators
_OPERAND_TOKEN = re.compile(r'[^\s,+\-*\[\]]+')

_MEM_OPERATORS = frozenset('+-*')


class X86Preprocessor(ASMPreprocessor):
    branch_instructions = BRANCH_INSTRUCTIONS

    def parse_operands(self, operands: str) -> Iterable[str]:
        offset = 0
        length = len(operands)

        while offset < length:
            match operands[offset]:
                case ' ' | ',':
                    offset += 1
                case '[':
                    end = operands.index(']', offset)
                    yield '['
                    yield from self.parse_memory_expression(operands[offset + 1 : end])
                    yield ']'
                    offset = end + 1
                case ch if ch in _MEM_OPERATORS:
                    yield ch
                    offset += 1
                case _:
                    m = _OPERAND_TOKEN.match(operands, offset)
                    if not m:
                        offset += 1
                        continue

                    token = m.group().rstrip(':')  # strip segment-override colon (e.g. "fs:")
                    lower = token.lower()

                    if lower in _SIZE_QUALIFIER_SET:
                        # consume the mandatory following "ptr" keyword and merge into one token
                        rest = operands[m.end() :].lstrip()
                        if rest.lower().startswith('ptr'):
                            yield f'{lower}_ptr'
                            offset = m.end() + (len(operands[m.end() :]) - len(rest)) + 3
                        else:
                            yield lower
                            offset = m.end()
                    else:
                        yield lower
                        offset = m.end()

    def parse_memory_expression(self, expr: str) -> Iterator[str]:
        """
        We need this because subtracting a memory expression in ghidra is done by [rbp + -0x8].
        Split a memory address expression into tokens.
        treating negative displacements like "-0x8" as a single token rather than an operator followed by a number.
        """
        expr = expr.strip()
        i = 0
        length = len(expr)

        while i < length:
            match expr[i]:
                case ' ':
                    i += 1
                case '+' | '*':
                    yield expr[i]
                    i += 1
                case '-':
                    # unary minus: attach to the number that follows when preceded by an operator or start
                    prev = expr[:i].rstrip()
                    if not prev or prev[-1] in ('+', '-', '*'):
                        m = _OPERAND_TOKEN.match(expr, i + 1)
                        if m:
                            yield f'-{m.group().lower()}'
                            i = m.end()
                            continue
                    yield '-'
                    i += 1
                case _:
                    m = _OPERAND_TOKEN.match(expr, i)
                    if m:
                        yield m.group().lower()
                        i = m.end()
                    else:
                        i += 1
