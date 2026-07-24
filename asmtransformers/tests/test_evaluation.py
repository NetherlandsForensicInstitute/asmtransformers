import random

import numpy as np
import pytest
from datasets import Dataset

from scripts.evaluation import calculate_all, calculate_one_rank, generate_anchor_pos_pairs


@pytest.fixture
def rank1():
    return {
        'anchor': {'embeddings': [0, 1, 2]},
        'pos': {'embeddings': [1, 2, 3]},
        'negs': np.array([[4, 5, 6], [7, 8, 9]]),
    }


@pytest.fixture
def rank2():
    return {
        'anchor': {'embeddings': [0, 1, 2]},
        'pos': {'embeddings': [4, 5, 6]},
        'negs': np.array([[7, 8, 9], [1, 2, 3]]),
    }


@pytest.fixture
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
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (1, 1)
    # one ranked first, one ranked second
    test_pools = [rank1, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (0.75, 1 / 2)
    # both ranked second place
    test_pools = [rank2, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (0.5, 0)


@pytest.fixture
def dataset():
    # same label means potential pair
    return Dataset.from_dict(
        {
            'label': ['return', 'return', 'jump', 'jump', 'move', 'move', 'add', 'add'],
            'cfg': [
                'ret',
                'ret',
                'j 0x32',
                'c.j 0x02',
                'c.mv a3,a5',
                'c.mv s8,a5',
                'c.addi4spn s0,sp,0x30',
                'addi a5,s0,-0xb0',
            ],
        }
    )


@pytest.fixture
def rng():
    return random.Random(10)


def test_generate_anchor_pos_pairs(dataset, rng):
    anchors, positives, anchor_labels, anchor_cfgs, pos_cfgs = generate_anchor_pos_pairs(dataset, rng, num_pairs=3)
    print(anchors)
    print(positives)
    assert len(anchors) == 3
    assert anchors[0] == {'label': 'add', 'cfg': 'addi a5,s0,-0xb0'}
    # make sure the anchors and positives of the same pair have the same index
    assert anchors[0]['label'] == 'add' and positives[0]['label'] == 'add'
    # assert pairs are always in the same order (this is more for continuity than for passing the test right now)
    assert anchors[0]['label'] == 'add' and anchors[1]['label'] == 'move' and anchors[2]['label'] == 'jump'
