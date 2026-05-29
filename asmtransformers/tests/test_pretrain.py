import json
import logging
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from scripts import pretrain as pretrain_module
from scripts.pretrain import (
    build_arg_parser,
    build_output_dir,
    build_training_args,
    configure_pretrain_logging,
    destroy_distributed_process_group,
    load_eval_dataset,
)


def build_training_args_kwargs(*, bf16=False, tf32=False):
    return {
        'output_dir': '/tmp/asmtransformers-test',
        'epoch': 1,
        'max_steps': 20,
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'save_steps': 30,
        'logging_steps': 5,
        'eval_dataset': [1],
        'learning_rate': 1e-4,
        'warmup_ratio': 0.06,
        'bf16': bf16,
        'tf32': tf32,
        'dataloader_num_workers': 4,
        'save_total_limit': 3,
        'seed': 123,
    }


@pytest.fixture(autouse=True)
def restore_root_logger():
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    yield
    for handler in root_logger.handlers:
        if handler not in original_handlers:
            handler.close()
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)


def handler_filenames():
    return {
        handler.baseFilename for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)
    }


def stream_handler_count():
    return sum(type(handler) is logging.StreamHandler for handler in logging.getLogger().handlers)


def write_test_config(tmp_path):
    config_path = tmp_path / 'config.json'
    config_path.write_text(
        json.dumps(
            {
                'vocab_size': 32,
                'hidden_size': 16,
                'num_hidden_layers': 1,
                'num_attention_heads': 2,
                'intermediate_size': 32,
                'max_position_embeddings': 8,
            }
        )
    )
    return config_path


def install_fake_pretrain_dependencies(monkeypatch, calls, *, world_process_zero=True, train_error=None):
    def save_tokenizer(output_dir):
        calls['tokenizer_output_dir'] = output_dir

    def load_tokenizer(tokenizer):
        calls['tokenizer_path'] = tokenizer
        return SimpleNamespace(save_pretrained=save_tokenizer)

    def build_model(config):
        shared_embedding = object()
        embeddings = SimpleNamespace(position_embeddings=shared_embedding, word_embeddings=shared_embedding)
        return SimpleNamespace(base_model=SimpleNamespace(embeddings=embeddings))

    def train(resume_from_checkpoint=None):
        calls['resume_from_checkpoint'] = resume_from_checkpoint
        if train_error is not None:
            raise train_error

    def build_trainer(*, model, args, data_collator, train_dataset, eval_dataset):
        calls['trainer_args'] = args
        calls['train_size'] = len(train_dataset)
        calls['eval_size'] = len(eval_dataset)
        return SimpleNamespace(
            is_world_process_zero=lambda: world_process_zero,
            train=train,
            save_model=lambda output_dir: calls.setdefault('model_output_dir', output_dir),
        )

    dataset = DatasetDict(
        {
            'train': Dataset.from_dict({'input_ids': [[1], [2]], 'attention_mask': [[1], [1]]}),
            'test': Dataset.from_dict({'input_ids': [[3]], 'attention_mask': [[1]]}),
        }
    )

    monkeypatch.setattr(pretrain_module, 'ASMTokenizer', SimpleNamespace(from_pretrained=load_tokenizer))
    monkeypatch.setattr(pretrain_module, 'ASMBertForMaskedLM', build_model)
    monkeypatch.setattr(pretrain_module, 'Trainer', build_trainer)
    monkeypatch.setattr(pretrain_module, 'load_from_disk', lambda path: dataset)
    monkeypatch.setattr(pretrain_module, 'DataCollatorForLanguageModeling', lambda **kwargs: object())


def run_test_pretrain(tmp_path, *, resume_from_checkpoint=None, run_id=None):
    pretrain_module.pretrain(
        model_path=None,
        output_dir=tmp_path,
        data='dataset-path',
        tokenizer='tokenizer-path',
        config=write_test_config(tmp_path),
        epoch=1,
        max_steps=-1,
        batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=3,
        logging_steps=1,
        mlm_prob=0.15,
        learning_rate=1e-4,
        warmup_ratio=0.06,
        bf16=False,
        tf32=False,
        dataloader_num_workers=0,
        save_total_limit=2,
        eval_samples=1,
        seed=42,
        resume_from_checkpoint=resume_from_checkpoint,
        run_id=run_id,
    )


def test_arg_parser_requires_output_dir():
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_positional_output_dir():
    args = build_arg_parser().parse_args(['output', '--data', 'dataset-path'])

    assert args.output_dir == 'output'
    assert args.data == 'dataset-path'


def test_arg_parser_accepts_run_id():
    args = build_arg_parser().parse_args(['output', '--data', 'dataset-path', '--run-id', 'manual'])

    assert args.run_id == 'manual'


def test_build_output_dir_uses_cli_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv('ASMTRANSFORMERS_RUN_ID', 'env-run')
    monkeypatch.setenv('SLURM_JOB_ID', '12345')

    assert build_output_dir(tmp_path, run_id='manual') == str(tmp_path / 'pretraining_mlm_manual')


def test_build_output_dir_uses_env_run_id_before_slurm_job_id(monkeypatch, tmp_path):
    monkeypatch.setenv('ASMTRANSFORMERS_RUN_ID', 'env-run')
    monkeypatch.setenv('SLURM_JOB_ID', '12345')

    assert build_output_dir(tmp_path) == str(tmp_path / 'pretraining_mlm_env-run')


def test_build_output_dir_uses_slurm_job_id(monkeypatch, tmp_path):
    monkeypatch.delenv('ASMTRANSFORMERS_RUN_ID', raising=False)
    monkeypatch.setenv('SLURM_JOB_ID', '12345')

    assert build_output_dir(tmp_path) == str(tmp_path / 'pretraining_mlm_slurm_12345')


def test_build_output_dir_uses_timestamp_without_run_id(monkeypatch, tmp_path):
    monkeypatch.delenv('ASMTRANSFORMERS_RUN_ID', raising=False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(pretrain_module, 'timestamp', lambda: '2026-05-22_12-00-00')

    assert build_output_dir(tmp_path) == str(tmp_path / 'pretraining_mlm_2026-05-22_12-00-00')


def test_configure_pretrain_logging_keeps_single_process_logging(monkeypatch, tmp_path):
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)

    configure_pretrain_logging(tmp_path)

    assert handler_filenames() == {str(tmp_path / 'training_logging.log')}
    assert stream_handler_count() == 1


def test_configure_pretrain_logging_adds_canonical_and_rank_log_for_rank_zero(monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')

    configure_pretrain_logging(tmp_path)

    assert handler_filenames() == {
        str(tmp_path / 'training_logging.log'),
        str(tmp_path / 'training_rank_0.log'),
    }
    assert stream_handler_count() == 1


def test_configure_pretrain_logging_uses_only_rank_log_for_nonzero_rank(monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')

    configure_pretrain_logging(tmp_path)

    assert handler_filenames() == {str(tmp_path / 'training_rank_1.log')}
    assert stream_handler_count() == 0


def test_build_training_args_rejects_bf16_without_cuda(monkeypatch):
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_available', lambda: False)

    with pytest.raises(RuntimeError, match='--bf16 requested, but CUDA is not available'):
        build_training_args(**build_training_args_kwargs(bf16=True))


def test_build_training_args_rejects_tf32_without_cuda(monkeypatch):
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_available', lambda: False)

    with pytest.raises(RuntimeError, match='--tf32 requested, but CUDA is not available'):
        build_training_args(**build_training_args_kwargs(tf32=True))


def test_build_training_args_rejects_bf16_when_cuda_device_does_not_support_it(monkeypatch):
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_bf16_supported', lambda: False)

    with pytest.raises(RuntimeError, match='does not support bfloat16'):
        build_training_args(**build_training_args_kwargs(bf16=True))


def test_build_training_args_rejects_tf32_before_ampere(monkeypatch):
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.cuda, 'get_device_capability', lambda: (7, 5))

    with pytest.raises(RuntimeError, match='compute capability 7.5; need 8.0\\+'):
        build_training_args(**build_training_args_kwargs(tf32=True))


def test_build_training_args_passes_supported_precision_through(monkeypatch):
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.cuda, 'get_device_capability', lambda: (8, 0))
    monkeypatch.setattr(pretrain_module, 'TrainingArguments', lambda **kwargs: SimpleNamespace(**kwargs))

    args = build_training_args(
        **build_training_args_kwargs(
            bf16=True,
            tf32=True,
        )
    )

    assert args.bf16 is True
    assert args.tf32 is True
    assert args.dataloader_num_workers == 4
    assert args.save_total_limit == 3
    assert args.seed == 123
    assert args.max_steps == 20
    assert args.eval_strategy == 'steps'
    assert args.eval_steps == 10


def test_load_eval_dataset_bounds_eval_samples():
    dataset = DatasetDict(
        {
            'train': Dataset.from_dict({'input_ids': [[1]], 'attention_mask': [[1]]}),
            'test': Dataset.from_dict({'input_ids': [[1], [2], [3]], 'attention_mask': [[1], [1], [1]]}),
        }
    )

    eval_dataset = load_eval_dataset(dataset, eval_samples=2)

    assert len(eval_dataset) == 2


@pytest.mark.parametrize(
    ('is_initialized', 'expected_destroyed'),
    [
        (True, True),
        (False, False),
    ],
)
def test_destroy_distributed_process_group(monkeypatch, is_initialized, expected_destroyed):
    calls = {}

    monkeypatch.setattr(pretrain_module.torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.distributed, 'is_initialized', lambda: is_initialized)
    monkeypatch.setattr(
        pretrain_module.torch.distributed, 'destroy_process_group', lambda: calls.setdefault('destroyed', True)
    )

    destroy_distributed_process_group()

    assert calls == ({'destroyed': True} if expected_destroyed else {})


def test_pretrain_passes_resume_checkpoint_to_trainer(monkeypatch, tmp_path):
    calls = {}

    install_fake_pretrain_dependencies(monkeypatch, calls)

    run_test_pretrain(tmp_path, resume_from_checkpoint='checkpoint-12')

    assert calls['tokenizer_path'] == 'tokenizer-path'
    assert calls['trainer_args'].bf16 is False
    assert calls['trainer_args'].save_total_limit == 2
    assert calls['train_size'] == 2
    assert calls['eval_size'] == 1
    assert calls['resume_from_checkpoint'] == 'checkpoint-12'
    assert calls['tokenizer_output_dir'].startswith(str(tmp_path))
    assert calls['model_output_dir'].startswith(str(tmp_path))


def test_pretrain_skips_tokenizer_save_on_nonzero_process(monkeypatch, tmp_path):
    calls = {}

    install_fake_pretrain_dependencies(monkeypatch, calls, world_process_zero=False)

    run_test_pretrain(tmp_path)

    assert 'tokenizer_output_dir' not in calls
    assert calls['model_output_dir'].startswith(str(tmp_path))


def test_pretrain_cleans_up_distributed_process_group_after_training_error(monkeypatch, tmp_path):
    calls = {}

    install_fake_pretrain_dependencies(monkeypatch, calls, train_error=RuntimeError('training failed'))
    monkeypatch.setattr(pretrain_module.torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(pretrain_module.torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(
        pretrain_module.torch.distributed, 'destroy_process_group', lambda: calls.setdefault('destroyed', True)
    )

    with pytest.raises(RuntimeError, match='training failed'):
        run_test_pretrain(tmp_path)

    assert calls['destroyed'] is True
    assert 'model_output_dir' not in calls
