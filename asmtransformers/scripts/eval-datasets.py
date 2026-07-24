import argparse
import random

from datasets import Dataset

from scripts.evaluation import generate_anchor_pos_pairs, generate_triplets, load_test_functions


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('data_folder', type=str, help='folder with input data')
    parser.add_argument('output_folder', type=str, help='folder to leave output data')
    parser.add_argument('--pool-size', type=int, help='the poolsize to pick the positive example from')
    parser.add_argument('--seed', type=int, default=4201, help='seed random evaluation sampling')

    return parser


def main(data_folder, output_folder, pool_size, seed):
    test_functions = load_test_functions(data_folder, architecture=None)
    anchor_rng = random.Random(seed)
    # todo: change num_pairs = 1000 before actually using the code
    anchor_pairs = generate_anchor_pos_pairs(test_functions, anchor_rng, num_pairs=300)
    test_pools = generate_triplets(test_functions, anchor_pairs, pool_size=pool_size, static_pool=True, rng=seed + 1)
    # todo: this does not work! find a way to save the data
    Dataset.from_generator(test_pools)
    Dataset.to_disk(output_folder)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
