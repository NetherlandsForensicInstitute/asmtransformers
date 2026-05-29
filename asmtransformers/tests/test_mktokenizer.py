import pytest
from datasets import Dataset

from asmtransformers.models import asmbert
from scripts.mktokenizer import extract_tokens_map

@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {
            'cfg': [
                '{"0": ["ret"]}',
                '{"1": ["add amd64"]}',
                '{"2": ["arm64"]}',
                '{"3": ["arm64"]}',
                '{"4": ["i386"]}',
                '{"5": ["i386"]}',
                '{"6": ["riscv64"]}',
            ],
            'architecture': [
                'amd64',
                'amd64',
                'arm64',
                'arm64',
                'i386',
                'i386',
                'riscv64',
            ],
        }
    )



def test_extract_tokens_map(dataset):
    empty_tokenizer = asmbert.ASMTokenizer(vocab_file=None)
    tokens = extract_tokens_map(empty_tokenizer, dataset)
    assert len(list(tokens)) == 6
    assert 'ret' in tokens
    assert type(tokens) == set
