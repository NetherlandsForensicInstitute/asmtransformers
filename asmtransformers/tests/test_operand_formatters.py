from asmtransformers import arm64, operands


def test_full_house():
    graph = {0x12: [
        'add x0,x0x,#0x1234',
        'sub w0,0x400,x1',
        'ldr x12,[sp, #-0x20]!',
        'mul w12,-0x123',
    ]}

    tokens = arm64.Preprocessor(operand_formatters=(
        operands.format_immediate_log,
        operands.format_offset_log,
    )).preprocess(graph)
    tokens_plain = arm64.Preprocessor().preprocess(graph)

    assert len(tokens) == len(tokens_plain)
    assert tokens != tokens_plain

    assert '#0x1234' not in tokens
    assert '#0x2^c' in tokens  # binary order of 0x1234 == 4660 == 2 ** 12 == 2^c
    assert '#0x1234' in tokens_plain

    assert '0x400' not in tokens
    assert '0x2^a' in tokens
    assert '0x400' in tokens_plain

    assert '#-0x20' not in tokens
    assert '#-0x20' in tokens_plain
    assert '#-0x2^5' in tokens  # binary order of 0x20 == 32 == 2 ** 5 == 2^5, keep sign

    assert '-0x123' not in tokens
    assert '-0x123' in tokens_plain
    assert '-0x2^8' in tokens
