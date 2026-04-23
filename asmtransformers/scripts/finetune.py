import argparse
import datetime as dt
import logging
import os
from pathlib import Path

import datasets
import torch
from sentence_transformers import LoggingHandler, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
from sentence_transformers.sentence_transformer.losses import (
    BatchHardTripletLossDistanceFunction,
    BatchSemiHardTripletLoss,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from tzlocal import get_localzone

from asmtransformers.datasets import as_sentence_transformer_training_dataset
from asmtransformers.models.asmsentencebert import ASMSentenceTransformer


def timestamp():
    return dt.datetime.now(tz=get_localzone()).strftime('%Y-%m-%d_%H-%M-%S')


def wrap_method(instance, method, new_method):
    """
    Wrap the specified method with another method that can inspect the paramters and original return
    value of the method and change the return value.

    :param instance: The instance that contains the method to wrap
    :param method: The method to wrap
    :param new_method: The wrapper method, of the form func(args, kwargs, return_value)
    """
    orig_method = getattr(instance, method)

    def wrapper(*args, **kwargs):
        result = orig_method(*args, **kwargs)
        return new_method(args, kwargs, result)

    setattr(instance, method, wrapper)


def make_train_dataset(functions):
    return as_sentence_transformer_training_dataset(functions)


def make_dev_evaluator(functions, batch_size):
    return TripletEvaluator(
        anchors=functions['anchor'],
        positives=functions['pos'],
        negatives=functions['neg'],
        batch_size=batch_size,
        name='dev',
    )


def get_mixed_precision_kwargs(use_amp):
    if not use_amp or not torch.cuda.is_available():
        return {'bf16': False, 'fp16': False}

    if torch.cuda.is_bf16_supported():
        return {'bf16': True, 'fp16': False}

    return {'bf16': False, 'fp16': True}


def main(data_folder, model, batch_size):
    """
    This script takes a language model and finetunes it for the task of semantic
    text similarity. This model, together with some logging and evaluation is dropped in
    an output folder

    :param data_folder: a folder with .jsonl.gz files in it. This is the training data
    :param model: the model to be finetuned
    :param batch_size: the batch size
    """

    model_name_or_path = model
    model_name = Path(model_name_or_path).stem if Path(model_name_or_path).is_dir() else model_name_or_path
    num_epochs = 3
    use_amp = torch.cuda.is_available()
    evaluation_steps = 50_000
    warmup_steps = 500

    # Save path of the model
    model_save_path = f'output/aarch64_ft_{model_name.replace("/", "-")}-{timestamp()}'
    Path(model_save_path).mkdir(exist_ok=True, parents=True)

    # Logging to a file
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler(), logging.FileHandler(filename=f'{model_save_path}/training_logging.log')],
    )

    mixed_precision_kwargs = get_mixed_precision_kwargs(use_amp)
    model = ASMSentenceTransformer.from_basemodel(
        base_model_name_or_path=model_name_or_path,
        model_args={'torch_dtype': torch.bfloat16 if mixed_precision_kwargs['bf16'] else torch.float32},
    )
    logging.info(f'pre-trained model {model_name} loaded')

    functions = datasets.load_from_disk(data_folder)
    train_functions = functions['train']
    logging.info('training data loaded')

    train_dataset = make_train_dataset(train_functions)
    test_functions = functions['test']
    dev_evaluator = make_dev_evaluator(test_functions, batch_size=batch_size)
    logging.info('training and evaluation data created')

    # Configure the training.
    # The jTrans loss is all triplets (including easy) with a cosine metric and a margin of 0.2 (BatchAllTripletLoss)
    # BatchSemiHardTripletLoss works on all non-easy triplets. That seems to give better training losses.
    train_loss = BatchSemiHardTripletLoss(
        model, distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance, margin=0.2
    )

    def log_loss(args, kwargs, result):
        # log it. Open and close the file to prevent loss.
        loss = result.item()
        with open(os.path.join(model_save_path, 'train_loss.log'), 'a') as loss_log:
            loss_log.write(f'{loss}\n')
        # Return the original result, unchanged.
        return result

    wrap_method(train_loss, 'batch_semi_hard_triplet_loss', log_loss)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=model_save_path,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        eval_strategy='steps',
        eval_steps=evaluation_steps,
        save_strategy='steps',
        save_steps=100_000 * 16 // batch_size,
        save_total_limit=0,
        report_to='none',
        **mixed_precision_kwargs,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=dev_evaluator,
        loss=train_loss,
        processing_class=model.tokenizer,
    )

    trainer.train()
    trainer.save_model(model_save_path)

    metrics = dev_evaluator(model, output_path=model_save_path)
    logging.info(f'Final evaluation metrics: {metrics}')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder', type=str, required=True, help='folder with data')

    parser.add_argument('-m', '--model', type=str, required=True, help='The name of the model used for finetuning')

    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        required=False,
        default=16,
        help='Feed the data to the model in batches for a potential speed-up',
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
