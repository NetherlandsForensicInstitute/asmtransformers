import json

import numpy as np
import pytest
import torch
from transformers import BertConfig

from asmtransformers.models.asmbert import ASMBertModel
from asmtransformers.models.asmsentencebert import batch_semi_hard_triplet_loss, build_finetuning_model
from asmtransformers.models.embedder import ASMEmbedder


@pytest.fixture
def checkpoint_path(tmp_path):
    vocab = [f'JUMP_ADDR_{index}' for index in range(512)] + [
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
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


def test_from_basemodel_freezes_jtrans_default_layers(checkpoint_path):
    model = build_finetuning_model(checkpoint_path)
    bert_model = model.model.base_model

    assert all(not param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[0].parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[1].parameters())


def test_from_basemodel_can_disable_freezing(checkpoint_path):
    model = build_finetuning_model(
        checkpoint_path,
        freeze_embeddings=False,
        freeze_layer_count=0,
    )
    bert_model = model.model.base_model

    assert all(param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(param.requires_grad for layer in bert_model.encoder.layer for param in layer.parameters())


def test_from_basemodel_freezes_configured_layer_count(checkpoint_path):
    model = build_finetuning_model(
        checkpoint_path,
        freeze_embeddings=False,
        freeze_layer_count=1,
    )
    bert_model = model.model.base_model

    assert all(param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[0].parameters())
    assert all(param.requires_grad for param in bert_model.encoder.layer[1].parameters())


def test_finetuning_model_can_be_saved_and_reloaded(checkpoint_path, tmp_path):
    model = build_finetuning_model(checkpoint_path)
    output_path = tmp_path / 'saved-model'

    model.save_pretrained(output_path)
    reloaded = ASMEmbedder.from_pretrained(output_path)
    embedding = reloaded.encode('[[4096, ["ret"]]]')

    assert (output_path / 'model.safetensors').is_file()
    assert (output_path / 'tokenizer.json').is_file()
    assert embedding.shape == (8,)
    assert embedding.dtype == np.float32


def test_batch_semi_hard_triplet_loss_is_zero_when_negative_satisfies_margin():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.2)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_batch_semi_hard_triplet_loss_is_positive_when_negative_is_close():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.99, 0.01],
        ]
    )
    labels = torch.tensor([0, 0, 1])

    loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.2)

    assert loss > 0


def test_batch_semi_hard_triplet_loss_handles_batches_without_valid_positives():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])

    loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.2)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_batch_semi_hard_triplet_loss_handles_batches_without_valid_negatives():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0])

    loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.2)

    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))
