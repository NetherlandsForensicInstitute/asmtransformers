import re
from collections.abc import Iterator


from asmtransformers.operands import is_offset

# find a way to find branch instructions for x86
#find a way to see if # is also used in x86 ghidra for immediate parts and otherwise find way to look for immediate in x86
# check for hexa and also if it is a branch target?
# check for size, we see qword ptr 


# immediates do not have # but look like 0x10 or -0x8. We also find hexadecimals like 0x4000
# MOV                 RDX, qword ptr [RBP + -0x8] => sb_local

BRANCH_INSTRUCTIONS = (
    # CALL calls external function that lives outside context
    'CALL',
    # Different jump calls
    'JNZ',
    'JMP',
    'JZ',
    'JNC',
    # Jump instructions found on https://en.wikipedia.org/wiki/List_of_x86_instructions
    'JA',
    'JAE',
    'JB',
    'JBE',
    'JC',
    'JG',
    'JGE',
    'JL',
    'JLE',
    'JNA',
    'JNAE',
    'JNB',
    'JNBE',
    'JNC',
    'JNE',
    'JNG',
    'JNGE',
    'JNL',
    'JNLE',
    'JNO',
    'JNP',
    'JNS',
    'JNZ',
    'JO',
    'JP',
    'JPE',
    'JNS',
    'JNZ',
    'JO',
    'JP',
    'JPE',
    'JPO',
    'JS',
    'JCXZ',
    'JECXZ',
    'JMP',
)
