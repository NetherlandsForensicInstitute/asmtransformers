from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Sequence
from typing import ClassVar

from asmtransformers.operands import Formatter, is_offset


class ASMPreprocessor(ABC):
    """
    Base class for architecture-specific preprocessing of assembly code.

    Subclasses should provide at least two things:

    - a `branch_instructions` class or instance attribute, used to determine whether offset operands should be
      translated into jump target tokens;
    - `parse_operands`, a method to split the operands part of an instruction into a token or tokens.

    Instructions strings are assumed to be "<instruction> <operand1> <operand2>". Should the decompilation result format
    this in a different way, subclasses could override `parse_instruction`, which should in turn call `parse_operands`
    with *all operands* in a single string.
    """

    branch_instructions: ClassVar[Collection[str]] = ()

    def __init__(
        self,
        *,
        context_length: int = 512,
        prefix_tokens: Sequence[str] = None,
        operand_formatters: Sequence[Formatter] = None,
    ):
        """

        Subclasses are encouraged to *not* override this.

        :param context_length: The maximum number of tokens to consider in scope. Jumps outside of this context length
            should be considered to exceed the context length and use the appropriate token.
        :param prefix_tokens: Tokens to insert before the first instruction (this also influences the minimum jump
            target).
        :param operand_formatters: An ordered sequence of formatters to optionally translate operands into a different
            representation (note that the first formatter that produces a replacement is used, subsequent formatters are
            ignored for that particular operand).
        """
        self.context_length = context_length
        self.prefix_tokens = prefix_tokens or ()
        self.operand_formatters = operand_formatters or ()

    def format_jump(self, operand: str, target_index: int | None) -> str:
        """
        Formats a target jump address into a jump token, either inside or outside the context length.

        :param operand: The original operand that refers to a jump address.
        :param target_index: The token index the operand refers to.
        :return: A jump token.
        """
        if target_index is None:
            return 'UNK_JUMP_ADDR'
        elif target_index < self.context_length:
            return f'JUMP_ADDR_{target_index}'
        else:
            return 'JUMP_ADDR_EXCEEDED'

    def format_operand(self, operand: str) -> str | None:
        """
        Queries the configured operand formatters in order whether the operand should be formatted.

        Note that the first formatter to update the operand is used blindly, without querying the remaining formatters.

        :param operand: The operand to be formatted.
        :return: An updated operand, or `None`.
        """
        for formatter in self.operand_formatters:
            # provide the first non-falsy value returned by any formatter
            if replacement := formatter(operand):
                return replacement
        # no format override, signal use as-is
        return None

    def parse_instruction(self, instruction: str) -> tuple[str, tuple[str, ...]]:
        """
        Splits `instruction` into the instruction mnemonic and its operands, as `(instruction, (token1, token2, ...))`.

        Note that the operands section of the resulting value need not match the common syntax for this, operand parsers
        for specific architectures can choose to separate `[x1]` into 3 tokens `[`, `x1` and `]`.

        :param instruction: the full instruction string to parse, e.g. `add x1 #0x01`.
        :return: `(token, (token, ...))`, where the first token represents the instruction mnemonic.
        """
        match instruction.lower().split(maxsplit=1):
            case [instruction]:
                # no operands
                return instruction, ()
            case instruction, operands:
                return instruction, tuple(self.parse_operands(operands))
            case _:
                raise ValueError(f'failed to parse instruction "{instruction}"')

    @abstractmethod
    def parse_operands(self, operands: str) -> Iterable[str]:
        """
        Architecture-specific parsing of operands for an instruction.

        The translation from any number of operands to any number of tokens is up to the architecture implementor.

        :param operands: operands to be parsed.
        :return: An iterable of tokens (e.g. a `list` or a `generator`).
        """
        ...

    def preprocess(self, function_blocks: dict[int, list[str]]) -> list[str]:
        """
        Main entrypoint for a preprocessor, translates a control flow graph into a list of tokens.

        NB: the control flow graph is sorted by address before processing it, assuming the entry point for the graph is
            the lowest address in the graph!

        :param function_blocks: A control flow graph encoded as a *{address → [instruction, ...]}* structure.
        :return: The list of tokens representing the provided control flow graph.
        """
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
                instruction, operands = self.parse_instruction(instruction)

                tokens.append(instruction)
                for operand in operands:
                    if instruction in self.branch_instructions and is_offset(operand):
                        # an operand to a branching instruction that is formatted as a hexadecimal number
                        # this is interpreter as the offset to a basic block, and this tracked as such in path_offsets
                        jump_target = int(operand, base=16)
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
