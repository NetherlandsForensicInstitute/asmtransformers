import argparse
import random

import tqdm
from datasets import Dataset
import os


def generate_train(data_folder, output_folder):
    train_data_folder = os.path.join(data_folder, 'train')
    train_output_folder = os.path.join(output_folder, 'train')

    print("Opening dataset")
    train_functions = Dataset.load_from_disk(train_data_folder)  # .select(range(100000))

    print("Creating category mapping")

    def label_name_mapper(example):
        label = example['bin_name'] + "/" + example['func_name']
        example['label_name'] = label
        return example

    train_functions = train_functions.map(label_name_mapper, batch_size=1000, num_proc=8)
    unique_labels = train_functions.unique('label_name')
    label2id = {label: i for i, label in enumerate(unique_labels)}
    print(label2id)

    print("Applying labels")

    def labeler(example):
        example['label'] = label2id[example['label_name']]
        return example

    train_functions = train_functions.map(labeler, batch_size=1000, num_proc=8)

    print("Saving...")
    train_functions.save_to_disk(train_output_folder)


def generate_eval(data_folder, output_folder):
    test_data_folder = os.path.join(data_folder, 'test')
    test_output_folder = os.path.join(output_folder, 'test')

    print("Opening dataset")
    test_functions = Dataset.load_from_disk(test_data_folder).select(range(1000000))
    print(test_functions)

    def add_random(example):
        label = example['bin_name'] + "/" + example['func_name']
        example['label'] = label
        # Ensure same labels are sorted together and in random order.
        example['label_random'] = f"{label}\0{random.random()}"
        return example

    print("Adding columns")
    # Don't use all 3M examples because sort is really slow.
    test_functions = test_functions.map(add_random, batch_size=10000, num_proc=8)
    print("Sorting dataset")
    test_functions = test_functions.sort('label')

    print("Sorting positives")
    # materialize it to disk because it's A LOT faster
    test_functions.save_to_disk('/data/temp/pos')
    test_functions_pos = Dataset.load_from_disk('/data/temp/pos')
    test_functions_pos = test_functions_pos.sort('label_random')

    print("Shuffling negatives")
    test_functions_neg = test_functions.shuffle()

    print("Building triplets")
    triplets = []
    for anchor, pos, neg in tqdm.tqdm(zip(test_functions, test_functions_pos, test_functions_neg)):
        if anchor['label'] != pos['label']:
            raise RuntimeError("mismatch")
        if anchor['label_random'] == pos['label_random']:
            # same entry, reject
            continue
        if anchor['cfg'] == pos['cfg']:
            # same content, reject
            continue
        if anchor['label'] == neg['label']:
            # same label, reject
            continue
        if anchor['cfg'] == neg['cfg']:
            # same content, reject
            continue
        triplets.append({'anchor': anchor['cfg'], 'pos': pos['cfg'], 'neg': neg['cfg']})
    print(f"We have {len(triplets)} usable triplets")
    trip = Dataset.from_list(triplets)
    trip.save_to_disk(test_output_folder)


def main(data_folder, output_folder):
    generate_train(data_folder, output_folder)
    generate_eval(data_folder, output_folder)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        required=True,
        help="folder with data"
    )
    parser.add_argument(
        '-o',
        '--output-folder',
        type=str,
        required=True,
        help="folder with data"
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
