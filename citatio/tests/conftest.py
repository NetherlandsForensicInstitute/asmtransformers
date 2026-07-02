import json
from pathlib import Path

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch


TEST_ROOT = Path(__file__).parent


@pytest.fixture(scope='session')
def monkeypatch_session():
    # regular monkeypatch fixture is function scope, provide a session-scoped one
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope='session')
def functions():
    return json.loads((TEST_ROOT / 'functions.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def embeddings(functions):
    arrays = np.load(TEST_ROOT / 'functions.npy', allow_pickle=False)
    return {function['name']: embedding for function, embedding in zip(functions, arrays, strict=True)}
