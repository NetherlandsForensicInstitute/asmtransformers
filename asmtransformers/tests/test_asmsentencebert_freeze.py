import json

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
from transformers import BertConfig

from asmtransformers.models.asmbert import ASMBertModel
from asmtransformers.models.asmsentencebert import build_finetuning_model


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
    bert_model = model[0].model.base_model

    assert all(not param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[0].parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[1].parameters())


def test_from_basemodel_can_disable_freezing(checkpoint_path):
    model = build_finetuning_model(
        checkpoint_path,
        freeze_embeddings=False,
        freeze_layer_count=0,
    )
    bert_model = model[0].model.base_model

    assert all(param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(param.requires_grad for layer in bert_model.encoder.layer for param in layer.parameters())


def test_from_basemodel_freezes_configured_layer_count(checkpoint_path):
    model = build_finetuning_model(
        checkpoint_path,
        freeze_embeddings=False,
        freeze_layer_count=1,
    )
    bert_model = model[0].model.base_model

    assert all(param.requires_grad for param in bert_model.embeddings.parameters())
    assert all(not param.requires_grad for param in bert_model.encoder.layer[0].parameters())
    assert all(param.requires_grad for param in bert_model.encoder.layer[1].parameters())


def test_finetuning_model_can_be_saved_and_reloaded(checkpoint_path, tmp_path):
    model = build_finetuning_model(checkpoint_path)
    output_path = tmp_path / 'saved-model'

    model.save(output_path, create_model_card=False)
    reloaded = SentenceTransformer(str(output_path))
    embedding = reloaded.encode('[[4096, ["ret"]]]')

    assert (output_path / 'modules.json').is_file()
    assert (output_path / '0_ASMTransformerModule' / 'model.safetensors').is_file()
    assert (output_path / '0_ASMTransformerModule' / 'tokenizer.json').is_file()
    assert embedding.shape == (8,)
    assert embedding.dtype == np.float32


def test_saving_finetuning_model_does_not_query_huggingface_for_base_model(checkpoint_path, tmp_path, monkeypatch):
    def fail_get_model_info(*args, **kwargs):
        raise AssertionError('unexpected Hugging Face model lookup')

    monkeypatch.setattr('sentence_transformers.base.model_card.get_model_info', fail_get_model_info)
    model = build_finetuning_model(checkpoint_path)

    model.save(tmp_path / 'saved-model')
