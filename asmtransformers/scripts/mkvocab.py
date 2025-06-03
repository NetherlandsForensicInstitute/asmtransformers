import json
from pathlib import Path

import datasets
from tqdm import tqdm

from asmtransformers import arm64, operands

DATASET = Path()  # Path to dataset, fill in yourself
OUTPUT = Path('./results/vocab.txt')

CONTEXT_LENGTH = 512
SAMPLE_SIZE = 0

if __name__ == '__main__':
    # use list comp to force ordering (lexicographic sort breaks without leaing zeroes)
    jump_targets = [f'JUMP_ADDR_{n}' for n in range(CONTEXT_LENGTH)]

    print('opening dataset ...')
    dataset = datasets.load_from_disk(DATASET)
    print('... done')
    preprocessor = arm64.Preprocessor(operand_formatters=(
        operands.format_immediate_log,
        operands.format_offset_log,
    ))
    tokens = set()

    if SAMPLE_SIZE:
        dataset = dataset.select(range(SAMPLE_SIZE))

    for subset in dataset:
        for sample in tqdm(dataset[subset], desc=f'processing {subset} subset'):
            function = dict(json.loads(sample['cfg']))
            tokens.update(preprocessor.preprocess(function))

    # remove the used jump targets from the collected tokens
    tokens -= set(jump_targets)

    with OUTPUT.open('wt') as output:
        for token in jump_targets:
            output.write(token)
            output.write('\n')
        for token in sorted(tokens):
            output.write(token)
            output.write('\n')
