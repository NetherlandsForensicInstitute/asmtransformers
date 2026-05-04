import re
from collections.abc import Callable, Iterator

from networkx import DiGraph

from asmtransformers.operands import is_offset


# based mostly on: https://en.wikipedia.org/wiki/List_of_x86_instructions
BRANCH_INSTRUCTIONS = (
    # unconditional jump and call/return
    'jmp',
    'call',
    'ret',
    'retn',
    'retf',
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


# Ghidra prefixes memory operands with a size qualifier followed by the keyword 'ptr'
# for example "dword ptr [rbp + -0x8]". We concat these two words into 1 mem token.
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


def parse_operands(operands: str) -> Iterator[str]:
    """
    move through x86 string operands linearly
    """
    offset = 0
    length = len(operands)

    while offset < length:
        match operands[offset]:
            case ' ' | ',':
                offset += 1
            case '[':
                end = operands.index(']', offset)
                yield '['
                yield from _parse_mem_expr(operands[offset + 1 : end])
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


def _parse_mem_expr(expr: str) -> Iterator[str]:
    """
    We need this because subtracting a memory expression in ghidra is done by [rbp + -0x8].
    Split a memory address expression into tokens.
    treating negative displacements like "-0x8" as a single token rather than an operator followed by a number."""
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


class X86Preprocessor:
    """
    Based on the ARM64 preprocessor but adjusted for amd64 (x86_64) arch.
    """

    def __init__(
        self,
        *,
        branch_instructions: tuple[str, ...] = BRANCH_INSTRUCTIONS,
        parse_operands: Callable[[str], Iterator[str]] = parse_operands,
        context_length: int = 512,
        prefix_tokens: tuple[str, ...] | None = None,
        operand_formatters: tuple[Callable, ...] | None = None,
    ):
        self.branch_instructions = frozenset(branch_instructions)
        self.parse_operands = parse_operands
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
            if replacement := formatter(operand):
                return replacement

    def preprocess(self, function_blocks: dict[int, list[str]]) -> list[str]:
        block_offsets = {}
        jump_offsets = {}
        tokens = list(self.prefix_tokens)

        if isinstance(function_blocks, DiGraph):
            function_blocks = {block_id: block['asm'] for block_id, block in sorted(function_blocks.nodes.items())}

        for block_id, block in function_blocks.items():
            block_offsets[block_id] = len(tokens)
            for instruction in block:
                parts = instruction.lower().split(maxsplit=1)
                mnemonic = parts[0]
                operand_str = parts[1] if len(parts) > 1 else ''

                tokens.append(mnemonic)
                for operand in self.parse_operands(operand_str) if operand_str else ():
                    if mnemonic in self.branch_instructions and is_offset(operand):
                        # can't slice at place 2 because negative hex values so therefore use value from is_offset regex
                        jump_target = int(is_offset(operand).group('value'), base=16)
                        jump_offsets[len(tokens)] = jump_target
                    else:
                        operand = self.format_operand(operand) or operand
                    tokens.append(operand)

        for offset, jump_target in jump_offsets.items():
            token = tokens[offset]
            if replacement := self.format_jump(token, block_offsets.get(jump_target)):
                tokens[offset] = replacement

        return tokens
