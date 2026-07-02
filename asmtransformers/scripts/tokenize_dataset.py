import argparse

import datasets

from asmtransformers.models.asmbert import ASMTokenizer


def tokenize(tokenizer, dataset, *, num_proc=10):
    def do_tokenize(function):
        encoded = tokenizer([function['cfg']], architecture=function.get('architecture', 'arm64'))
        return {key: value[0].tolist() for key, value in encoded.items()}

    map_kwargs = {'batched': False}
    if num_proc is not None:
        map_kwargs['num_proc'] = num_proc

    return dataset.map(do_tokenize, **map_kwargs)


def main(tokenizer, input_data, output_folder, split):
    tokenizer = ASMTokenizer.from_pretrained(tokenizer)
    print('loading dataset')
    dataset = datasets.load_from_disk(input_data)
    if split:
        dataset = dataset.train_test_split(test_size=split)

    # let the tokenizer preprocess data from data_in, write the result to data_out
    print('tokenizing dataset')
    if isinstance(dataset, datasets.Dataset):  # datasets.load_from_disk either a Dataset or DatasetDict type
        dataset = tokenize(tokenizer, dataset)
    else:
        for subset in dataset:
            dataset[subset] = tokenize(tokenizer, dataset[subset])

    dataset.save_to_disk(output_folder)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_data', type=str, help='data to be used for training')
    parser.add_argument('output_folder', type=str, help='folder to leave the tokenized data')
    parser.add_argument('tokenizer', type=str, help='folder with tokenizer')
    parser.add_argument(
        '--split',
        type=float,
        required=False,
        help='split between train and test; define percentage of test data as number between 0 and 1',
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
