import json

import numpy as np
import pytest
import torch
from transformers import BertConfig

from asmtransformers.models.asmbert import ASMBertModel
from asmtransformers.models.embedder import ASMEmbedder


@pytest.fixture
def cfg():
    return json.dumps([[4096, ['mov x0,#0x0', 'ret']]])


@pytest.fixture
def checkpoint_path(tmp_path):
    vocab = [f'JUMP_ADDR_{index}' for index in range(512)] + [
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
        'mov',
        'x0',
        '#0x0',
        'ret',
    ]
    (tmp_path / 'vocab.txt').write_text('\n'.join(vocab))
    (tmp_path / 'tokenizer_config.json').write_text(
        json.dumps(
            {
                'do_lower_case': False,
                'do_basic_tokenize': False,
                'tokenize_chinese_chars': False,
                'tokenizer_class': 'ARM64Tokenizer',
                'model_max_length': 512,
            }
        )
    )

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=vocab.index('[PAD]'),
    )
    model = ASMBertModel(config)
    model.save_pretrained(tmp_path)
    return tmp_path


def test_native_embedder_returns_normalized_embedding(checkpoint_path, cfg):
    embedder = ASMEmbedder.from_pretrained(checkpoint_path)

    embedding = embedder.encode(cfg)

    assert embedding.shape == (8,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_native_embedder_batch_size_does_not_change_embeddings(checkpoint_path, cfg):
    embedder = ASMEmbedder.from_pretrained(checkpoint_path)

    single = embedder.encode(cfg)
    batched = embedder.encode([cfg, cfg], batch_size=2)

    assert np.allclose(single, batched[0])
    assert np.allclose(single, batched[1])


def test_mean_pool_ignores_padding():
    token_embeddings = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 200.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]])

    pooled = ASMEmbedder.mean_pool(token_embeddings, attention_mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 3.0]]))
