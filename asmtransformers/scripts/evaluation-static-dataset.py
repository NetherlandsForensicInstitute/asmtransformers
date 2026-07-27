import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

from asmtransformers.models.embedder import ASMEmbedder
from scripts.evaluation import calculate_all, timestamp


def load_triplets(data_path):
    with Path(data_path).open('rb') as file:
        anchors, positives, neg_pools = pickle.load(file)
    anchors = model.encode(anchors)
    positives = model.encode(positives)
    neg_pools = model.encode(neg_pools)

    for i in range(len(anchors)):
        yield {
            'anchor': {'embeddings': anchors[i]},
            'pos': {'embeddings': positives[i]},
            'negs': neg_pools,
        }


def get_parser():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument(
        'input_path',
        type=Path,
        help='either the path to the test functions; or to folder containing anchor/pos/neg datasets '
        '(one subfolder for each architecture and one generic called "all"; requires --dataset-ready flag)',
    )
    parser.add_argument('output_path', type=Path, help='the path to write the final scores to')
    parser.add_argument('model_path', type=str, help='path to model')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_path = Path(args.model_path)

    model = ASMEmbedder.from_pretrained(model_path)

    model_name = model_path.parents[1].name
    output_file = f'{timestamp()}-{model_name}'

    results_per_architecture = defaultdict(int)
    architectures = ['arm64', 'amd64', 'riscv64', 'i386', 'all']
    for architecture in architectures:
        print(f'evaluating {architecture}')
        # load data and turn into usable triplets
        test_pools = load_triplets(Path(input_path, architecture, 'eval_data.pkl'))

        # calculate mrr & accuracy
        final_mrr, final_acc = calculate_all(test_pools, output_path, output_file + f'-{architecture}')
        results_per_architecture[architecture] = (final_mrr, final_acc)

    with Path(output_path, output_file + '-eval_per_architecture.csv').open('w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                'model_name',
                'mrr_arm64',
                'acc_arm64',
                'mrr_amd64',
                'acc_amd64',
                'mrr_riscv64',
                'acc_riscv64',
                'mrr_i386',
                'acc_i386',
                'mrr_all',
                'acc_all',
            )
        )
        writer.writerow(
            (
                model_name,
                results_per_architecture['arm64'][0],
                results_per_architecture['arm64'][1],
                results_per_architecture['amd64'][0],
                results_per_architecture['amd64'][1],
                results_per_architecture['riscv64'][0],
                results_per_architecture['riscv64'][1],
                results_per_architecture['i386'][0],
                results_per_architecture['i386'][1],
                results_per_architecture['all'][0],
                results_per_architecture['all'][1],
            )
        )

    with Path(args.output_path, output_file + '-parameters.txt').open('w') as file:
        file.write('\n'.join(f'{key}={value!s}' for key, value in vars(args).items()))
