import sys

from datasets import Dataset

from asmtransformers.models.asmbert import ARM64Tokenizer


def preprocess(tokenizer, dataset, *, num_proc=10):
    def tokenize(function):
        encoded = tokenizer([function['cfg']])
        return {key: value[0].tolist() for key, value in encoded.items()}

    map_kwargs = {'batched': False}
    if num_proc is not None:
        map_kwargs['num_proc'] = num_proc

    return dataset.map(tokenize, **map_kwargs)


if __name__ == '__main__':
    # expect 3 arguments from cli
    tokenizer, data_in, data_out = sys.argv[1:]

    # let the tokenizer preprocess data from data_in, write the result to data_out
    preprocess(
        ARM64Tokenizer.from_pretrained(tokenizer),
        Dataset.load_from_disk(data_in),
    ).save_to_disk(data_out)
