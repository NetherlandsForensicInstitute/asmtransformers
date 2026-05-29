import json
import sys
from pathlib import Path

import datasets
from tqdm import tqdm

from asmtransformers.models import asmbert


OUTPUT = Path('./results')

CONTEXT_LENGTH = 512
SAMPLE_SIZE = 0


def extract_tokens_map(tokenizer, dataset, subset_name='all'):

    def extract(cfgs, architectures):
        tokens = set()
        print(cfgs, architectures)
        for cfg, architecture in zip(cfgs, architectures):
            function = dict(json.loads(cfg))
            tokens.update(tokenizer.preprocessors[architecture].preprocess(function))
        return {'tokens': [list(tokens)]}

    all_tokens = set()
    # goal is to parallellize the process for speed purposes
    # the means is to make sub-datasets that we can process in parallel
    # this is done by making a new dataset where rows are lists of tokens that can later be joined to one vocab set
    token_dataset = dataset.map(extract, batched=True, batch_size=10000, remove_columns=dataset.column_names, input_columns=['cfg', 'architecture'], keep_in_memory=True, num_proc=4)
    for tokens in token_dataset['tokens']:
        all_tokens.update(tokens)
    return all_tokens


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
            tokens.update(extract_tokens_map(empty_tokenizer, dataset[subset], subset))
    else:
        tokens.update(extract_tokens_map(empty_tokenizer, dataset))

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
