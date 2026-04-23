import pytest
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel

from asmtransformers.models.asmbert import ASMBertForMaskedLM, ASMBertModel


@pytest.fixture(scope='function')
def model():
    # The model is created with a fixed seed to ensure that the same initialization weights are
    # generated for every run. It makes the test deterministic and eliminates
    # the (already vanishingly small) chance of a random initialization causing a test failure.
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=8,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return ASMBertForMaskedLM(config)


@pytest.fixture(scope='function')
def base_model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=8,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return ASMBertModel(config)


def test_position_and_word_embeddings_are_tied(model):
    word_embeddings = model.base_model.embeddings.word_embeddings
    position_embeddings = model.base_model.embeddings.position_embeddings

    assert position_embeddings is word_embeddings
    assert position_embeddings.weight is word_embeddings.weight
    assert position_embeddings.weight.data_ptr() == word_embeddings.weight.data_ptr()


def test_forward_reports_losses_and_jtp_ignores_out_of_range_labels(model):
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    # The two sets of labels have the same labels/targets for jump targets (<8),
    # and different labels/targets for non-jump targets (>=8).
    labels = torch.tensor([[7, 9, -100, 31]])
    labels_with_different_non_jtp_targets = torch.tensor([[7, 10, -100, 30]])

    output = model(input_ids=input_ids, labels=labels)
    output_with_different_non_jtp_targets = model(input_ids=input_ids, labels=labels_with_different_non_jtp_targets)

    assert output.loss is not None
    assert output.masked_lm_loss is not None
    assert output.jtp_loss is not None
    # Total loss is the sum of the masked language model loss and the jump target prediction (JTP) loss.
    assert torch.isclose(output.loss, output.masked_lm_loss + output.jtp_loss)
    # The two sets of labels have the same JTP targets, so the JTP loss should be the same.
    assert torch.isclose(output.jtp_loss, output_with_different_non_jtp_targets.jtp_loss)
    # The two sets of labels have different non-JTP targets. With these inputs MLM loss should be different.
    assert not torch.isclose(output.masked_lm_loss, output_with_different_non_jtp_targets.masked_lm_loss)


def test_training_step_updates_shared_embedding_weights(model):
    model.train()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, 2, 3, -100]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    shared_weight_before = model.base_model.embeddings.word_embeddings.weight.detach().clone()
    optimizer.zero_grad()

    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()

    shared_weight = model.base_model.embeddings.word_embeddings.weight

    assert shared_weight.grad is not None
    assert torch.count_nonzero(shared_weight.grad).item() > 0

    optimizer.step()

    position_weight = model.base_model.embeddings.position_embeddings.weight

    assert position_weight is shared_weight
    assert position_weight.data_ptr() == shared_weight.data_ptr()
    assert not torch.equal(shared_weight_before, shared_weight.detach())


def test_pickle_save_and_load_preserves_tied_embeddings(tmp_path, model):
    original_word_embeddings = model.base_model.embeddings.word_embeddings
    original_position_embeddings = model.base_model.embeddings.position_embeddings

    assert original_position_embeddings is original_word_embeddings
    assert original_position_embeddings.weight is original_word_embeddings.weight

    model.save_pretrained(tmp_path, safe_serialization=False)
    reloaded_model = ASMBertForMaskedLM.from_pretrained(tmp_path)

    reloaded_word_embeddings = reloaded_model.base_model.embeddings.word_embeddings
    reloaded_position_embeddings = reloaded_model.base_model.embeddings.position_embeddings

    assert reloaded_position_embeddings is reloaded_word_embeddings
    assert reloaded_position_embeddings.weight is reloaded_word_embeddings.weight


def test_safe_serialization_preserves_tied_embeddings(tmp_path, model):
    original_word_embeddings = model.base_model.embeddings.word_embeddings
    original_position_embeddings = model.base_model.embeddings.position_embeddings

    assert original_position_embeddings is original_word_embeddings
    assert original_position_embeddings.weight is original_word_embeddings.weight

    model.save_pretrained(tmp_path)
    reloaded_model = ASMBertForMaskedLM.from_pretrained(tmp_path)

    reloaded_word_embeddings = reloaded_model.base_model.embeddings.word_embeddings
    reloaded_position_embeddings = reloaded_model.base_model.embeddings.position_embeddings

    assert reloaded_position_embeddings is reloaded_word_embeddings
    assert reloaded_position_embeddings.weight is reloaded_word_embeddings.weight


def test_base_model_safe_serialization_preserves_tied_embeddings(tmp_path, base_model):
    original_word_embeddings = base_model.embeddings.word_embeddings
    original_position_embeddings = base_model.embeddings.position_embeddings

    assert original_position_embeddings is original_word_embeddings
    assert original_position_embeddings.weight is original_word_embeddings.weight

    base_model.save_pretrained(tmp_path)
    reloaded_model = ASMBertModel.from_pretrained(tmp_path)

    reloaded_word_embeddings = reloaded_model.embeddings.word_embeddings
    reloaded_position_embeddings = reloaded_model.embeddings.position_embeddings

    assert reloaded_position_embeddings is reloaded_word_embeddings
    assert reloaded_position_embeddings.weight is reloaded_word_embeddings.weight


def test_base_model_load_reports_no_missing_or_unexpected_keys(tmp_path, base_model):
    base_model.save_pretrained(tmp_path)

    _, loading_info = ASMBertModel.from_pretrained(tmp_path, output_loading_info=True)

    assert loading_info['missing_keys'] == set()
    assert loading_info['unexpected_keys'] == set()
    assert loading_info['mismatched_keys'] == set()


def test_base_model_loads_legacy_checkpoint_with_shared_embedding_only_in_position_key(tmp_path, base_model):
    config = base_model.config
    config.save_pretrained(tmp_path)

    state_dict = base_model.state_dict()
    shared_weight = state_dict['embeddings.word_embeddings.weight'].clone()
    del state_dict['embeddings.word_embeddings.weight']
    state_dict['embeddings.position_embeddings.weight'] = shared_weight
    save_file(state_dict, tmp_path / 'model.safetensors')

    reloaded_model, loading_info = ASMBertModel.from_pretrained(tmp_path, output_loading_info=True)

    reloaded_word_embeddings = reloaded_model.embeddings.word_embeddings.weight.detach()
    reloaded_position_embeddings = reloaded_model.embeddings.position_embeddings.weight.detach()

    assert loading_info['missing_keys'] == set()
    assert loading_info['unexpected_keys'] == set()
    assert loading_info['mismatched_keys'] == set()
    assert torch.equal(reloaded_word_embeddings, shared_weight)
    assert torch.equal(reloaded_position_embeddings, shared_weight)


def test_base_model_warns_when_legacy_compatibility_path_is_used(monkeypatch, base_model):
    delayed_model = ASMBertModel(base_model.config, delay_tie_for_load=True)
    shared_weight = base_model.embeddings.word_embeddings.weight.detach().clone()
    delayed_model.embeddings.position_embeddings.weight.data.copy_(shared_weight)

    loading_info = {
        'missing_keys': {'embeddings.word_embeddings.weight'},
        'unexpected_keys': set(),
        'mismatched_keys': set(),
        'error_msgs': [],
    }

    @classmethod
    def fake_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert kwargs['output_loading_info'] is True
        assert kwargs['delay_tie_for_load'] is True
        return delayed_model, loading_info

    monkeypatch.setattr(BertModel, 'from_pretrained', fake_from_pretrained)

    with pytest.warns(UserWarning, match='Loaded legacy ASMBertModel checkpoint'):
        reloaded_model, reloaded_info = ASMBertModel.from_pretrained('legacy-model', output_loading_info=True)

    assert reloaded_info['missing_keys'] == set()
    assert torch.equal(reloaded_model.embeddings.word_embeddings.weight.detach(), shared_weight)
    assert reloaded_model.embeddings.position_embeddings is reloaded_model.embeddings.word_embeddings
