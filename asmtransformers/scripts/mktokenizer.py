import json
import sys
from pathlib import Path

import datasets
from tqdm import tqdm

from asmtransformers.models import asmbert


OUTPUT = Path('./results')

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

    # initialize "empty" tokenizer, to be able to use each architecture's preprocessors
    # in order to create a vocabulary of this dataset
    empty_tokenizer = asmbert.ASMTokenizer(vocab_file=None)
    tokens = set()

    if 'train' in dataset:
        for subset in dataset:
            tokens.update(extract_tokens(empty_tokenizer, dataset[subset], subset))
    else:
        tokens.update(extract_tokens(empty_tokenizer, dataset))

    # remove the used jump targets from the collected tokens
    tokens -= set(jump_targets)

    token_ids = {}

    for id_, token in enumerate(jump_targets):
        token_ids[token] = id_

    for id_, token in enumerate(tokens, start=CONTEXT_LENGTH):
        token_ids[token] = id_

    real_tokenizer = asmbert.ASMTokenizer(vocab_file=None, vocab=token_ids)
    real_tokenizer.save_pretrained(OUTPUT)


if __name__ == '__main__':
    mkvocab(Path(sys.argv[1]))
