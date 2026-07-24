import random

import numpy as np
import pytest
from datasets import Dataset

from scripts.evaluation import calculate_all, calculate_one_rank, generate_anchor_pos_pairs, generate_neg_pool


@pytest.fixture
def rank1():
    # pos [1, 2, 3] is closer to anchor [0, 1, 2] than negs, therefore it is ranked 1
    return {
        'anchor': {'embeddings': [0, 1, 2]},
        'pos': {'embeddings': [1, 2, 3]},
        'negs': np.array([[4, 5, 6], [7, 8, 9]]),
    }


@pytest.fixture
def rank2():
    # pos [4, 5, 6] is closer to anchor [0, 1, 2] than [7, 8, 9], but further than [1, 2, 3]
    # therefore the positive is ranked 2nd
    return {
        'anchor': {'embeddings': [0, 1, 2]},
        'pos': {'embeddings': [4, 5, 6]},
        'negs': np.array([[7, 8, 9], [1, 2, 3]]),
    }


@pytest.fixture
# pos [7, 8, 9] is further away from anchor [0, 1, 2] than both negs, and is therefore ranked 3rd/last
def rank3():
    return {
        'anchor': {'embeddings': [0, 1, 2]},
        'pos': {'embeddings': [7, 8, 9]},
        'negs': np.array([[4, 5, 6], [1, 2, 3]]),
    }


def test_calculate_one_rank(rank1, rank2, rank3):
    assert calculate_one_rank(rank1) == 1
    assert calculate_one_rank(rank2) == 2
    assert calculate_one_rank(rank3) == 3


def test_calculate_all(rank1, rank2, tmp_path):
    # both ranked first
    test_pools = [rank1, rank1]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all_1.csv') == (1, 1)
    # one ranked first, one ranked second
    test_pools = [rank1, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all_0_75.csv') == (0.75, 1 / 2)
    # both ranked second place
    test_pools = [rank2, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all_0_5.csv') == (0.5, 0)


@pytest.fixture
def dataset():
    # same label means potential pair
    return Dataset.from_dict(
        {
            'label': ['return', 'return', 'jump', 'jump', 'move', 'move', 'add', 'add'],
            # these are not full cfgs, but single instructions. this is a simplification for testing's sake
            'cfg': [
                '[[0, [ret]]]',
                '[[0, [ret]]]',
                '[[1, [j 0x32]]]',
                '[[2, [c.j 0x02]]]',
                '[[3, [c.mv a3,a5]]]',
                '[[4, [c.mv s8,a5]]]',
                '[[5, [c.addi4spn s0,sp,0x30]]]',
                '[[6, [addi a5,s0,-0xb0]]]',
            ],
        }
    )


@pytest.fixture
def rng():
    return random.Random(10)


@pytest.fixture
def pos_anchor_pairs(dataset, rng):
    anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs = generate_anchor_pos_pairs(dataset, rng, num_pairs=3)
    return anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs


def test_generate_anchor_pos_pairs(pos_anchor_pairs):
    anchors, positives, _, _, _ = pos_anchor_pairs
    assert len(anchors) == 3
    assert anchors[0] == {'label': 'add', 'cfg': '[[6, [addi a5,s0,-0xb0]]]'}
    # make sure the anchors and positives of the same pair have the same index
    assert anchors[0]['label'] == 'add' and positives[0]['label'] == 'add'
    # assert pairs are always in the same order (this is more for continuity than for passing the test right now)
    assert anchors[0]['label'] == 'add' and anchors[1]['label'] == 'move' and anchors[2]['label'] == 'jump'


@pytest.fixture
def dataset_extended():
    return Dataset.from_dict(
        {
            'label': ['a', 'b', 'c', 'd', 'e', 'jump'],
            # these are not full cfgs, but single instructions. this is a simplification for testing's sake
            'cfg': [
                '[[7, [rax, rbx]]]',
                '[[8, [rsp,-0x10]]]',
                '[[9, [rbp+-0x8]]]',
                '[[10, [dword ptr [rbp + -0x8]]]]',
                '[[11, [xmmword ptr [rax]]]]',
                # this one will be rejected based on label & cfgs being part of the anchor pos pairs
                '[[12, [j 0x32]]]',
            ],
            'embeddings': [1, 2, 3, 4, 5, 6],
        }
    )


def test_generate_neg_pool(pos_anchor_pairs, dataset_extended, rng):
    _, _, anchor_labels, anchor_cfgs, pos_cfgs = pos_anchor_pairs
    # we can't request for a pool size 6, because there are only 5 valid negatives in dataset_extended
    with pytest.raises(ValueError):
        generate_neg_pool(6, dataset_extended, anchor_labels, anchor_cfgs, pos_cfgs, rng)
    neg_embeddings = generate_neg_pool(5, dataset_extended, anchor_labels, anchor_cfgs, pos_cfgs, rng)
    # once again we check that neg embeddings are always in the same order
    assert neg_embeddings[0] == 2 and neg_embeddings[1] == 3 and neg_embeddings[-1] == 1
