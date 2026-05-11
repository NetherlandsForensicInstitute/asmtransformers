from collections.abc import Sequence
from abc import ABC, abstractmethod
from typing import ClassVar


class ASMPreprocessor(ABC):
    branch_instructions: ClassVar[Sequence[str]] = ()

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

    @abstractmethod
    def parse_instruction(self, instruction: str) -> tuple[str, tuple[str, ...]]:
        ...

    @abstractmethod
    def is_offset(self, operand: str) -> bool:
        ...

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
                instruction, operands = self.parse_instruction(instruction)

                tokens.append(instruction)
                for operand in operands:
                    if instruction in self.branch_instructions and self.is_offset(operand):
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
