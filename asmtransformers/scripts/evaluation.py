import argparse
import csv
import datetime as dt
import os.path
import random
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

import datasets
import numpy as np
from scipy import spatial
from tqdm import tqdm
from tzlocal import get_localzone


def timestamp():
    return dt.datetime.now(tz=get_localzone()).strftime('%Y-%m-%d_%H-%M-%S')


def add_label(example):
    """create a label by concatenating the binary name and the function name

    :param example: row in dataset
    :returns example with an extra column containing the label"""
    label = example['file_name'] + '/' + example['function_name']
    example['label'] = label
    # Ensure same labels are sorted together and in random order.
    example['label_random'] = f'{label}\0{random.random()}'
    return example


def generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs, rng):
    """
    Generate a pool that does not contain the labels and cfgs in the anchors/pos

    :param pool_size: The number of items to add to the pool
    :param dataset: the dataset we draw the items from
    :param anchor_labels: item labels to avoid
    :param anchor_cfgs: item cfgs to avoid
    :param pos_cfgs: item cfgs to avoid
    :param rng: random number generator
    :return: a numpy array containing the embeddings of the items in the pool
    """
    neg_embeddings = []
    candidate_indices = list(range(len(dataset)))
    rng.shuffle(candidate_indices)
    for index in candidate_indices:
        neg = dataset[index]
        if neg['label'] in anchor_labels:
            # same label, reject
            continue
        if neg['cfg'] in anchor_cfgs:
            # same content, reject
            continue
        if neg['cfg'] in pos_cfgs:
            # same content, reject
            continue
        neg_embeddings.append(neg['embeddings'])
        if len(neg_embeddings) == pool_size:
            break
    if len(neg_embeddings) < pool_size:
        raise ValueError(f'only {len(neg_embeddings)} eligible negative examples available for pool_size={pool_size}')
    return np.array(neg_embeddings)


def generate_anchor_pos_pairs(dataset, rng):
    """generates anchor/positive pairs while rejecting identical CFGs."""
    labels = dataset['label']
    labels_with_indices = list(enumerate(labels))
    labels_with_indices.sort(key=itemgetter(1))

    label2index = {}
    for label, tuples in tqdm(groupby(labels_with_indices, key=itemgetter(1)), desc='building anchor/pos pairs'):
        label2index[label] = [index for index, label in tuples]

    anchors = []
    positives = []
    anchor_labels = set()
    anchor_cfgs = set()
    pos_cfgs = set()
    rejected = 0

    # If you are not evaluating on the entire data set (which we are not, because it's so big),
    # then you need to make sure the sample used is representative. Because of the mechanics of picking the
    # positive/negatives, we can't just subsample the dataset first.
    # We use the entire dataset, but we generate random anchors/positves/negatives drawn from the
    # entire set instead.

    while len(anchors) < 1000:  # Should take about an hour
        # Pick a random label
        label = rng.choice(labels)
        indexes = label2index[label]
        if len(indexes) < 2:
            # Not enough examples
            continue

        index_anchor, index_pos = rng.sample(indexes, 2)
        anchor = dataset[index_anchor]
        pos = dataset[index_pos]

        # jTrans eval does not reject identical inputs for pos
        # We do. Our dataset contains a large number of identical CFGs for functions.
        # This could be due to difference in compilers/ISA between x86_64 and ARM64.
        if anchor['cfg'] == pos['cfg']:
            # same content, reject
            rejected += 1
            continue

        anchors.append(anchor)
        positives.append(pos)
        anchor_labels.add(anchor['label'])
        anchor_cfgs.add(anchor['cfg'])
        pos_cfgs.add(pos['cfg'])

    return anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs


def generate_triplets(dataset, anchor_pairs, pool_size, static_pool, rng):
    """generates triplets from fixed anchor/positive pairs and sampled negative pools.

    :param dataset: huggingface dataset
    :param anchor_pairs: anchor/positive rows and exclusion sets
    :param pool_size: number of negative samples in triplets
    :param static_pool: keep or regenerate the negative pool for every anchor-pos pair.
    :param rng: random number generator
    :return: Generator containing triplets: anchor, pos, numpy.array([neg_embedding ...])
    """
    anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs = anchor_pairs

    if static_pool:
        pool = generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs, rng)
        for i in range(len(anchors)):
            yield {'anchor': anchors[i], 'pos': positives[i], 'negs': pool}
    else:
        for i in range(len(anchors)):
            # Generate a new negatives pool for every anchor/pos pair
            pool = generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs, rng)
            yield {'anchor': anchors[i], 'pos': positives[i], 'negs': pool}


def load_test_functions(data_folder, architecture=None):
    dataset = datasets.load_from_disk(data_folder)  # .select(range(11000, 45000))
    if architecture:
        print(f'Selecting examples with architecture=={architecture}')
        dataset = dataset.filter(lambda x: x['architecture'] == architecture)

    print('Adding columns')
    # Don't use all 3M examples because sort is really slow.
    test_functions = dataset.map(add_label, batch_size=10000, num_proc=8)
    print('Sorting dataset')
    return test_functions.sort('label')


def calculate_one_rank(row):
    """calculate and rank similarities between anchor function compilation and pos and pool
    :param row: anchor, pos, and list of negs
    :return The rank of the positive example (pos) in the pool.
    """
    anchor, pos, negs = row['anchor']['embeddings'], row['pos']['embeddings'], row['negs']

    # calculate the cosine distance between the anchor and pos
    cosine_sim_pos = 1 - spatial.distance.cosine(anchor, pos)
    # calculate the cosine distance between the anchor and negs
    anchor_norm = np.linalg.norm(anchor)
    neg_norms = np.linalg.norm(negs, axis=1, keepdims=True)
    similarities = (anchor @ negs.T / (anchor_norm * neg_norms.T)).T
    similarities = np.sort(np.squeeze(similarities))[::-1]
    for i, sim in enumerate(similarities):
        if cosine_sim_pos >= sim:
            # Our rank is equal to the number of items that are better than us, plus one.
            return i + 1

    # If we fall through the for loop without hitting the return statement,
    # we are worse than all the negatives. Our rank is therefore
    # equal to the number of negatives.
    return len(similarities)


def calculate_all(test_pools, output_path, output_file):
    """calculate the mean reciprocal rank and precision@1 of the specified test pools
    :param test_pools: Dataset unit that yields anchor, pos, and list of numpy array of neg embeddings
    :param output_path: Folder where the results are stored
    :param output_file: name of the file where the results are stored.
    """
    sum_rr = 0
    sum_acc = 0
    final_mrr = 0.0
    final_acc = 0.0
    with open(os.path.join(output_path, output_file + '-results.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('iteration', 'MRR', 'P@1'))
        for i, row in enumerate(tqdm(test_pools)):
            rank = calculate_one_rank(row)
            rr = 1.0 / rank
            sum_rr += rr
            if rr == 1.0:
                sum_acc += 1.0
            final_mrr = sum_rr / (i + 1)
            final_acc = sum_acc / (i + 1)
            row_result = (i, final_mrr, final_acc)
            writer.writerow(row_result)
            csvfile.flush()
    return final_mrr, final_acc


def run_tests(data_folder, output_path, pool_size, static_pool, architecture, seed, repeats=1):
    if repeats < 1:
        raise ValueError('repeats must be at least 1')
    if repeats > 1 and not static_pool:
        raise ValueError('repeats greater than 1 are only supported with --static-pool')
    # architecture is a filter; if we want to evaluate all architectures at once, we do not filer the dataset
    if architecture == 'all':
        architecture = None

    print('\ngenerate test_pools\n')
    test_functions = load_test_functions(data_folder, architecture)
    anchor_rng = random.Random(seed)
    anchor_pairs = generate_anchor_pos_pairs(test_functions, anchor_rng)

    model_name = data_folder.split('/')[-1]
    output_file = f'{timestamp()}-{model_name}-{architecture}-{pool_size}-{static_pool}'
    repeat_seeds = [None] * repeats if seed is None else [seed + repeat + 1 for repeat in range(repeats)]
    aggregate_rows = []

    for repeat, repeat_seed in enumerate(repeat_seeds):
        pool_rng = random.Random(repeat_seed)
        test_pools = generate_triplets(
            test_functions, anchor_pairs, pool_size=pool_size, static_pool=static_pool, rng=pool_rng
        )
        print('\ncalculate cosine similarities\n')
        repeat_output_file = output_file if repeats == 1 else f'{output_file}-repeat-{repeat}'
        final_mrr, final_acc = calculate_all(test_pools, output_path, repeat_output_file)
        aggregate_rows.append((repeat, repeat_seed, final_mrr, final_acc))

    if repeats > 1:
        mrrs = np.array([row[2] for row in aggregate_rows])
        accuracies = np.array([row[3] for row in aggregate_rows])
        with open(os.path.join(output_path, output_file + '-aggregate.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(('repeat', 'seed', 'MRR', 'P@1'))
            writer.writerows(aggregate_rows)
            writer.writerow(('mean', '', float(np.mean(mrrs)), float(np.mean(accuracies))))
            writer.writerow(('std', '', float(np.std(mrrs)), float(np.std(accuracies))))
            writer.writerow(('min', '', float(np.min(mrrs)), float(np.min(accuracies))))
            writer.writerow(('max', '', float(np.max(mrrs)), float(np.max(accuracies))))

    return aggregate_rows


def get_parser():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('input_path', type=str, help='the path to the test data')
    parser.add_argument('output_path', type=str, help='the path to write the final scores to')
    parser.add_argument('--pool-size', type=int, help='the poolsize to pick the positive example from')
    parser.add_argument('--seed', type=int, default=4201, help='seed random evaluation sampling')
    parser.add_argument('--repeats', type=int, default=1, help='number of static-pool evaluation repeats')
    parser.add_argument(
        '--static-pool', action='store_true', help='keep the negatives pool or refresh for every anchor-pos pair'
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.input_path.split('/')[-1]
    output_file = f'{timestamp()}-{model_name}-{args.pool_size}-{args.static_pool}'
    results_per_architecture = defaultdict(int)
    architectures = ['arm64', 'amd64', 'riscv64', 'i386', 'all']
    for architecture in architectures:
        print(f'evaluating {architecture}')
        try:
            aggregate_rows = run_tests(
                args.input_path,
                args.output_path,
                args.pool_size,
                args.static_pool,
                architecture,
                args.seed,
                args.repeats,
            )
            # right now we only report the MRRs and not the accuracies as they are usually zero
            mrrs = np.array([row[2] for row in aggregate_rows])
            accs = np.array([row[3] for row in aggregate_rows])
            # if there are multiple repeats, there will be multiple mrrs so we take the mean of those
            results_per_architecture[architecture] = (np.mean(mrrs), np.mean(accs))

        except ValueError as error:
            parser.error(str(error))

    with open(os.path.join(args.output_path, output_file + '-eval_per_architecture.csv'), 'w') as csvfile:
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

    with open(os.path.join(args.output_path, output_file + '-parameters.txt'), 'w') as file:
        file.write(
            f'{args.input_path=},\n {args.output_path=},\n {args.pool_size=},\n {args.static_pool=},\n'
            f' {args.seed=},\n {args.repeats=},\n'
        )
