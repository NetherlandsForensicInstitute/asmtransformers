import argparse
import csv
import datetime
import os.path
from itertools import groupby
from operator import itemgetter
import random

import datasets
import numpy as np

from scipy import spatial

from tqdm import tqdm


def add_label(example):
    """create a label by concatenating the binary name and the function name

    :param example: row in dataset
    :returns example with an extra column containing the label"""
    label = example['bin_name'] + "/" + example['func_name']
    example['label'] = label
    # Ensure same labels are sorted together and in random order.
    example['label_random'] = f"{label}\0{random.random()}"
    return example


def generate_single_neg(dataset):
    """chooses a random row from the dataset

    :param dataset: huggingface dataset
    :return: random row from the dataset"""
    return dataset[random.randint(0, len(dataset) - 1)]


def generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs):
    """
    Generate a pool that does not contain the labels and cfgs in the anchors/pos

    :param pool_size: The number of items to add to the pool
    :param dataset: the dataset we draw the items from
    :param anchor_labels: item labels to avoid
    :param anchor_cfgs: item cfgs to avoid
    :param pos_cfgs: item cfgs to avoid
    :return: a numpy array containing the embeddings of the items in the pool
    """
    neg_embeddings = []
    while len(neg_embeddings) < pool_size:
        neg = generate_single_neg(dataset)
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
    return np.array(neg_embeddings)


def generate_triplets(dataset, pool_size, static_pool):
    """generates triplets consisting of an anchor, a positive example and a pool of negative examples, while making
    sure that none of the negative examples contain the same cfg as the positive and anchor they are linked to.
    Also check that the anchor and pos are not exactly the same.
    Negative samples are chosen by randomly choosing a row from the dataset.

    :param dataset: huggingface dataset
    :param pool_size: number of negative samples in triplets
    :param static_pool: keep or regenerate the negative pool for every anchor-pos pair.
    :return: Generator containing triplets: anchor, pos, numpy.array([neg_embedding ...])
    """
    labels = dataset['label']
    labels_with_indices = list(enumerate(labels))
    labels_with_indices.sort(key=itemgetter(1))

    label2index = {}
    for label, tuples in tqdm(groupby(labels_with_indices, key=itemgetter(1)),
                              desc='building anchor/pos pairs'):
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
        label = random.choice(labels)
        indexes = label2index[label]
        if len(indexes) < 2:
            # Not enough examples
            continue

        index_anchor, index_pos = random.sample(indexes, 2)
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

    if static_pool:
        pool = generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs)
        for i in range(len(anchors)):
            yield {'anchor': anchors[i], 'pos': positives[i], 'negs': pool}
    else:
        for i in range(len(anchors)):
            # Generate a new negatives pool for every anchor/pos pair
            pool = generate_neg_pool(pool_size, dataset, anchor_labels, anchor_cfgs, pos_cfgs)
            yield {'anchor': anchors[i], 'pos': positives[i], 'negs': pool}


def generate_test_pools(data_folder, pool_size, static_pool):
    """order data in such a way that we can make triplets consisting of an anchor, a positive and pool_size * negative
    examples, then call generate_triplets() to generate said triplets
    :param data_folder: Path to data
    :param pool_size: number of negative items to compare with
    :param static_pool: use the same pool of negatives for every pos/anchor pair (faster)

    return
    Generator yielding triplets: anchor, positive, numpy.array(negative_embeddings * POOL_SIZE)"""
    dataset = datasets.load_from_disk(data_folder)  # .select(range(11000, 45000))

    print("Adding columns")
    # Don't use all 3M examples because sort is really slow.
    test_functions = dataset.map(add_label, batch_size=10000, num_proc=8)
    print("Sorting dataset")
    test_functions = test_functions.sort('label')
    yield from generate_triplets(dataset=test_functions, pool_size=pool_size, static_pool=static_pool)


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
    with open(os.path.join(output_path, output_file + '-results.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('iteration', 'MRR', 'P@1'))
        for i, row in enumerate(test_pools):
            rank = calculate_one_rank(row)
            rr = 1.0 / rank
            sum_rr += rr
            if rr == 1.0:
                sum_acc += 1.0
            row_result = (i, sum_rr / (i + 1), sum_acc / (i + 1))
            print(row_result)
            writer.writerow(row_result)
            csvfile.flush()


def run_tests(data_folder, output_path, pool_size, static_pool):
    print('\ngenerate test_pools\n')
    test_pools = generate_test_pools(data_folder, pool_size, static_pool=static_pool)
    print('\ncalculate cosine similarities\n')
    model_name = data_folder.split('/')[-1]
    output_file = f"{model_name}-{pool_size}-{static_pool}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    calculate_all(test_pools, output_path, output_file)
    with open(os.path.join(output_path, output_file + '-parameters.txt'), 'w') as file:
        file.write(f"{data_folder=},\n {output_path=},\n {pool_size=},\n {static_pool=}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--input-path", type=str,
                        help='the path to the test data')
    parser.add_argument("--output-path", type=str,
                        help='the path to write the final scores to')
    parser.add_argument("--pool-size", type=int,
                        help='the poolsize to pick the positive example from')
    parser.add_argument("--static-pool", action='store_true',
                        help='keep the negatives pool or refresh for every anchor-pos pair')

    args = parser.parse_args()
    run_tests(args.input_path, args.output_path, args.pool_size, args.static_pool)
