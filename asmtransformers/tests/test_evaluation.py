import numpy as np
import pytest

from scripts import evaluation
from scripts.evaluation import (
    calculate_all,
    calculate_one_rank,
    generate_triplets,
    generate_anchor_pos_pairs,
    generate_neg_pool
)

@pytest.fixture
def rank1():
    return {'anchor':{'embeddings':[0, 1, 2]},
           'pos': {'embeddings':[1, 2, 3]},
           'negs':np.array([[4, 5, 6], [7, 8, 9]])}

@pytest.fixture
def rank2():
    return {'anchor': {'embeddings': [0, 1, 2]},
         'pos': {'embeddings': [4, 5, 6]},
         'negs': np.array([[7, 8, 9], [1, 2, 3]])}

def test_calculate_one_rank(rank1, rank2):
    assert calculate_one_rank(rank1) == 1
    assert calculate_one_rank(rank2) == 2


def test_calculate_all(rank1, rank2, tmp_path):
    # both ranked first
    test_pools = [rank1, rank1]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (1, 1)
    # one ranked first, one ranked second
    test_pools = [rank1, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (0.75, 1/2)
    # both ranked second place
    test_pools = [rank2, rank2]
    assert calculate_all(test_pools, tmp_path, 'test_calculate_all.csv') == (0.5, 0)