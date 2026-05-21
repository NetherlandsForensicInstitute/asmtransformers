import argparse
import datetime as dt
import logging
from importlib import resources
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import BertConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tzlocal import get_localzone

from asmtransformers.models import model_resource
from asmtransformers.models.asmbert import ASMBertForMaskedLM, ASMTokenizer


def timestamp():
    return dt.datetime.now(tz=get_localzone()).strftime('%Y-%m-%d_%H-%M-%S')


def validate_precision_support(*, bf16, tf32):
    if not bf16 and not tf32:
        return

    if not torch.cuda.is_available():
        requested = ', '.join(flag for flag, enabled in (('--bf16', bf16), ('--tf32', tf32)) if enabled)
        raise RuntimeError(f'{requested} requested, but CUDA is not available')

    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('--bf16 requested, but the current CUDA device does not support bfloat16')

    if tf32:
        capability = torch.cuda.get_device_capability()
        if capability < (8, 0):
            version = '.'.join(str(part) for part in capability)
            raise RuntimeError(
                f'--tf32 requested, but the current CUDA device has compute capability {version}; need 8.0+'
            )


def build_training_args(
    *,
    output_dir,
    epoch,
    max_steps,
    batch_size,
    gradient_accumulation_steps,
    save_steps,
    logging_steps,
    eval_dataset,
    learning_rate,
    warmup_ratio,
    bf16,
    tf32,
    dataloader_num_workers,
    save_total_limit,
    seed,
):
    validate_precision_support(bf16=bf16, tf32=tf32)
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        warmup_ratio=warmup_ratio,
        num_train_epochs=epoch,
        max_steps=max_steps,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps=max(1, save_steps // 3),
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=True,
        report_to=['tensorboard'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=bf16,
        tf32=tf32,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
    )


def load_eval_dataset(functions, eval_samples):
    if 'test' not in functions:
        return None

    eval_tokenized = functions['test'].shuffle(seed=42)
    if eval_samples is not None:
        eval_tokenized = eval_tokenized.select(range(min(eval_samples, len(eval_tokenized))))
    return eval_tokenized


def pretrain(
    model_path,
    output_dir,
    data,
    tokenizer,
    config,
    epoch,
    max_steps,
    batch_size,
    gradient_accumulation_steps,
    save_steps,
    logging_steps,
    mlm_prob,
    learning_rate,
    warmup_ratio,
    bf16,
    tf32,
    dataloader_num_workers,
    save_total_limit,
    eval_samples,
    seed,
    resume_from_checkpoint,
):
    output_dir = f'{output_dir}/pretraining_mlm_{timestamp()}'
    validate_precision_support(bf16=bf16, tf32=tf32)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(filename=f'{output_dir}/training_logging.log')],
    )

    logging.info('Script used: pretraining with MLM')
    logging.info(f'Saving checkpoints to: {output_dir}')

    # Load the tokenizer and model
    default_model_resource = model_resource('multilingual_asmbert')
    if tokenizer is None:
        with resources.as_file(default_model_resource) as tokenizer_path:
            tokenizer = ASMTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = ASMTokenizer.from_pretrained(tokenizer)

    if model_path:
        model = ASMBertForMaskedLM.from_pretrained(model_path)
        assert model.base_model.embeddings.position_embeddings is model.base_model.embeddings.word_embeddings
    else:
        # if no model_path is given, initialise a 'clean ASMBert'
        if config is None:
            config_resource = default_model_resource.joinpath('config.json')
            with resources.as_file(config_resource) as config_path:
                config = BertConfig.from_json_file(config_path)
        else:
            config = BertConfig.from_json_file(config)
        model = ASMBertForMaskedLM(config)
        assert model.base_model.embeddings.position_embeddings is model.base_model.embeddings.word_embeddings
    logging.info('Tokenizer and model loaded')

    # Load the training and evaluation datasets
    functions = load_from_disk(data)
    train_tokenized = functions['train'].shuffle(seed=42)
    eval_tokenized = load_eval_dataset(functions, eval_samples)
    logging.info('Datasets loaded')

    # given our tokenization, there's no sense in differentiating between whole word and token masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

    training_args = build_training_args(
        output_dir=output_dir,
        epoch=epoch,
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_dataset=eval_tokenized,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        tf32=tf32,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=save_total_limit,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    logging.info(f'Save tokenizer to: {output_dir}')
    tokenizer.save_pretrained(output_dir)

    logging.info('Start training')
    logging.info(
        f'Model parameters: epochs={epoch}, eval_steps={save_steps}, batch_size={batch_size}, '
        f'gradient_accumulation_steps={gradient_accumulation_steps}, mlm_prob={mlm_prob}, bf16={bf16}, tf32={tf32}'
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logging.info(f'Save model to: {output_dir}')
    model.save_pretrained(output_dir)

    logging.info('Training done')


def build_arg_parser():
    parser = argparse.ArgumentParser(description='ASM-Pretrain')
    parser.add_argument(
        'output_dir',
        type=str,
        help='the directory where the pretrained model will be saved',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='the path of the model to pretrain, can be empty if you want to initialise a new model',
    )
    parser.add_argument('--data', type=str, help='training dataset')
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='the path of tokenizer; defaults to the packaged multilingual_asmbert tokenizer',
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='the path of the model config used when initializing a new model. Defaults to packaged '
        'multilingual_asmbert',
    )
    parser.add_argument('--epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max-steps', type=int, default=-1, help='maximum number of training steps; -1 uses epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--save-steps', type=int, default=10000, help='after how many steps evaluate and save model')
    parser.add_argument('--logging-steps', type=int, default=100, help='number of update steps between two logs')
    parser.add_argument('--mlm-prob', type=float, default=0.15, help='probability of a token/word to be masked')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.06, help='warmup ratio for the learning-rate scheduler')
    parser.add_argument('--bf16', action='store_true', help='enable CUDA bfloat16 mixed precision training')
    parser.add_argument('--tf32', action='store_true', help='enable TF32 matmul/cudnn on supported CUDA GPUs')
    parser.add_argument(
        '--dataloader-num-workers',
        type=int,
        default=0,
        help='number of worker processes used by each training dataloader',
    )
    parser.add_argument('--save-total-limit', type=int, default=5, help='maximum number of checkpoints to keep')
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=100_000,
        help='maximum number of test samples used for intermediate evaluation; use -1 to disable the limit',
    )
    parser.add_argument('--seed', type=int, default=42, help='training seed')
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='path to a Trainer checkpoint to resume from',
    )
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.eval_samples < 0:
        args.eval_samples = None

    pretrain(**vars(args))
