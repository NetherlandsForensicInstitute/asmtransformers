import json
from pathlib import Path

import numpy as np
import pytest


TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope='session')
def functions():
    return json.loads((TEST_ROOT / 'functions.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def embeddings(functions):
    arrays = np.load(TEST_ROOT / 'functions.npy', allow_pickle=False)
    return {function['name']: embedding
            for function, embedding in zip(functions, arrays)}
