import re
from collections.abc import Iterator


from asmtransformers.operands import is_offset

# find a way to find branch instructions for x86
#find a way to see if # is also used in x86 ghidra for immediate parts and otherwise find way to look for immediate in x86
# check for hexa and also if it is a branch target?
# check for size 
