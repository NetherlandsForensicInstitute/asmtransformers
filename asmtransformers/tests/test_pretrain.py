import json
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from scripts import pretrain as pretrain_module
from scripts.pretrain import build_training_args, load_eval_dataset


def test_build_training_args_enables_bf16_without_requiring_cuda():
    args = build_training_args(
        output_dir='/tmp/asmtransformers-test',
        epoch=1,
        max_steps=20,
        batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=30,
        logging_steps=5,
        eval_dataset=[1],
        learning_rate=1e-4,
        warmup_ratio=0.06,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        save_total_limit=3,
        seed=123,
    )

    assert args.bf16 is True
    assert args.tf32 is None
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


def test_pretrain_passes_resume_checkpoint_to_trainer(monkeypatch, tmp_path):
    calls = {}

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            calls['tokenizer_output_dir'] = output_dir

    class FakeBertTokenizer:
        @classmethod
        def from_pretrained(cls, tokenizer):
            calls['tokenizer_path'] = tokenizer
            return FakeTokenizer()

    class FakeModel:
        def __init__(self, config):
            shared_embedding = object()
            self.base_model = SimpleNamespace(
                embeddings=SimpleNamespace(
                    position_embeddings=shared_embedding,
                    word_embeddings=shared_embedding,
                )
            )

        def save_pretrained(self, output_dir):
            calls['model_output_dir'] = output_dir

    class FakeTrainer:
        def __init__(self, *, model, args, data_collator, train_dataset, eval_dataset):
            calls['trainer_args'] = args
            calls['train_size'] = len(train_dataset)
            calls['eval_size'] = len(eval_dataset)

        def train(self, resume_from_checkpoint=None):
            calls['resume_from_checkpoint'] = resume_from_checkpoint

    dataset = DatasetDict(
        {
            'train': Dataset.from_dict({'input_ids': [[1], [2]], 'attention_mask': [[1], [1]]}),
            'test': Dataset.from_dict({'input_ids': [[3]], 'attention_mask': [[1]]}),
        }
    )
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

    monkeypatch.setattr(pretrain_module, 'BertTokenizer', FakeBertTokenizer)
    monkeypatch.setattr(pretrain_module, 'ASMBertForMaskedLM', FakeModel)
    monkeypatch.setattr(pretrain_module, 'Trainer', FakeTrainer)
    monkeypatch.setattr(pretrain_module, 'load_from_disk', lambda path: dataset)
    monkeypatch.setattr(pretrain_module, 'DataCollatorForLanguageModeling', lambda **kwargs: object())

    pretrain_module.pretrain(
        model_path=None,
        output_dir=tmp_path,
        data='dataset-path',
        tokenizer='tokenizer-path',
        config=config_path,
        epoch=1,
        max_steps=-1,
        batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=3,
        logging_steps=1,
        mlm_prob=0.15,
        learning_rate=1e-4,
        warmup_ratio=0.06,
        bf16=True,
        tf32=True,
        dataloader_num_workers=0,
        save_total_limit=2,
        eval_samples=1,
        seed=42,
        resume_from_checkpoint='checkpoint-12',
    )

    assert calls['tokenizer_path'] == 'tokenizer-path'
    assert calls['trainer_args'].bf16 is True
    assert calls['trainer_args'].save_total_limit == 2
    assert calls['train_size'] == 2
    assert calls['eval_size'] == 1
    assert calls['resume_from_checkpoint'] == 'checkpoint-12'
