import json
import sys
from pathlib import Path

import datasets
from tqdm import tqdm

from asmtransformers.models import asmbert


OUTPUT = Path('./results/vocab.txt')

CONTEXT_LENGTH = 512
SAMPLE_SIZE = 0


def extract_tokens(tokenizer, dataset, subset_name='all'):
    tokens = set()

    if SAMPLE_SIZE:
        dataset = dataset.select(range(SAMPLE_SIZE))

    for sample in tqdm(dataset, desc=f'processing {subset_name}'):
        function = dict(json.loads(sample['cfg']))
        # Use the architecture-specific preprocessor to process the function into tokens
        tokens.update(tokenizer.preprocessors[sample['architecture']].preprocess(function))

    return tokens


def mkvocab(dataset_file):
    # use list comp to force ordering (lexicographic sort breaks without leaing zeroes)
    jump_targets = [f'JUMP_ADDR_{n}' for n in range(CONTEXT_LENGTH)]

    print('opening dataset ...')
    dataset = datasets.load_from_disk(dataset_file)
    print('... done')

    # Open without a vocab, the entire point is to make a new one.
    tokenizer = asmbert.ASMTokenizer('/dev/null')
    tokens = set()

    if 'train' in dataset:
        for subset in dataset:
            tokens.update(extract_tokens(tokenizer, dataset[subset], subset))
    else:
        tokens.update(extract_tokens(tokenizer, dataset))

    # remove the used jump targets from the collected tokens
    tokens -= set(jump_targets)

    with OUTPUT.open('wt') as output:
        for token in jump_targets:
            output.write(token)
            output.write('\n')
        for token in sorted(tokens):
            output.write(token)
            output.write('\n')


if __name__ == '__main__':
    mkvocab(Path(sys.argv[1]))
