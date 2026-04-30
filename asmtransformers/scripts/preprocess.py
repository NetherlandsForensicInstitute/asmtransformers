import sys

import datasets

from asmtransformers.models.asmbert import ASMTokenizer


def preprocess(tokenizer, dataset, *, num_proc=10):
    def tokenize(function):
        encoded = tokenizer([function['cfg']], architecture=function.get('architecture', 'arm64'))
        return {key: value[0].tolist() for key, value in encoded.items()}

    map_kwargs = {'batched': False}
    if num_proc is not None:
        map_kwargs['num_proc'] = num_proc

    return dataset.map(tokenize, **map_kwargs)


if __name__ == '__main__':
    # expect 3 arguments from cli
    tokenizer, data_in, data_out = sys.argv[1:]

    tokenizer = ASMTokenizer.from_pretrained(tokenizer)
    dataset = datasets.load_from_disk(data_in)

    # let the tokenizer preprocess data from data_in, write the result to data_out
    for subset in dataset:
        dataset[subset] = preprocess(tokenizer, dataset[subset])

    dataset.save_to_disk(data_out)
