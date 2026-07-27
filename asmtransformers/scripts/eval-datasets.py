import argparse
import os
import pickle
import random
from pathlib import Path

from scripts.evaluation import generate_anchor_pos_pairs, generate_triplets, load_test_functions, generate_neg_pool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('data_folder', type=str, help='folder with input data')
    parser.add_argument('output_folder', type=str, help='folder to leave output data')
    parser.add_argument('--pool-size', type=int, help='the poolsize to pick the positive example from')
    parser.add_argument('--seed', type=int, default=4201, help='seed random evaluation sampling')

    return parser


def main(data_folder, output_folder, pool_size, seed):
    anchor_rng = random.Random(seed)
    architectures = ['arm64', 'amd64', 'riscv64', 'i386', 'all']
    for architecture in architectures:
        print(f'creating eval data for {architecture}')
        test_functions = load_test_functions(data_folder, architecture)
        # todo: change num_pairs = 1000 before actually using the code
        anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs = generate_anchor_pos_pairs(test_functions, anchor_rng, num_pairs=300)
        neg_pools = generate_neg_pool(pool_size, test_functions, anchor_labels, anchor_cfgs, pos_cfgs, random.Random(seed + 1))
        Path.mkdir(Path(output_folder, architecture), parents=True, exist_ok=True)
        with Path(output_folder, architecture, 'eval_data.pkl').open('wb') as f:
            pickle.dump((anchors, positives, neg_pools), f)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
