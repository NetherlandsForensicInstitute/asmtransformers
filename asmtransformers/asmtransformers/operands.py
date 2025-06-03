import re
from math import log2


def format_immediate_log(operand: str, threshold: int = 16) -> str | None:
    if (match := is_immediate(operand)) and (value := int(match.group('value'), 16)) > threshold:
        # value over threshold, mark the value as somewhere around its power of 2 (though marked with ^ instead
        # of **, as ** looks confusingly like 2 characters are hidden)
        # 4096 would be 2 ** 12, so #0x2^c, so would 6000, but 9000 would be #0x2^d
        return f'#{match.group("sign")}0x2^{int(log2(value)):x}'


def format_offset_log(operand: str) -> str | None:
    if match := is_offset(operand):
        value = int(match.group('value'), 16)
        if value > 0:
            # value is without sign, but 0 won't log
            value = int(log2(value))
        return f'{match.group("sign")}0x2^{value:x}'


def is_immediate(operand: str) -> re.Match | None:
    return re.match(r'#(?P<sign>[-+]?)0x(?P<value>[0-9a-f]+)', operand, re.IGNORECASE)


def is_offset(operand: str) -> re.Match | None:
    return re.match(r'(?P<sign>[-+]?)0x(?P<value>[0-9a-f]+)', operand, re.IGNORECASE)
