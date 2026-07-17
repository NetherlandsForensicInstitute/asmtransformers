import argparse
import datetime as dt
import logging
import numbers
import os
from pathlib import Path

import datasets
import torch
from sentence_transformers import InputExample, LoggingHandler, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from torch.utils.data import DataLoader
from tqdm import tqdm
from tzlocal import get_localzone

from asmtransformers.datasets import LazySentenceLabelDataset
from asmtransformers.models.asmsentencebert import build_finetuning_model


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


class OldFitScalarEvaluator:
    """Adapter for SentenceTransformers 5.x evaluators used with old_fit."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __call__(self, *args, **kwargs):
        # SentenceTransformers 5.x evaluators return metric dicts, while old_fit compares a scalar best score.
        # This adapter calls the underlying evaluator and returns the scalar score specified as "primary_metric"

        result = self.evaluator(*args, **kwargs)
        if isinstance(result, numbers.Number):
            return result

        if not isinstance(result, dict):
            raise RuntimeError(f'Expected evaluator to return a number or metrics dict, got {type(result).__name__}')

        primary_metric = getattr(self.evaluator, 'primary_metric', None)
        if primary_metric is None:
            raise RuntimeError('Evaluator returned a metrics dict without setting primary_metric')

        if primary_metric not in result:
            raise RuntimeError(f'Evaluator primary metric {primary_metric!r} not found in metrics: {sorted(result)}')

        return result[primary_metric]

    def __getattr__(self, name):
        return getattr(self.evaluator, name)


def collate_tokenized_batch(batch):
    features = {
        'input_ids': torch.tensor([example['input_ids'] for example in batch], dtype=torch.long),
        'attention_mask': torch.tensor([example['attention_mask'] for example in batch], dtype=torch.long),
    }
    labels = torch.tensor([example['label'] for example in batch], dtype=torch.long)
    return [features], labels


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
    num_epochs = 24
    use_amp = True  # Set to False, if you use a CPU or your GPU does not support FP16 operations
    evaluation_steps = 50_000
    warmup_steps = 500

    # Save path of the model
    model_save_path = f'output/asmtransformers_ft_{model_name.replace("/", "-")}-{timestamp()}'
    Path(model_save_path).mkdir(exist_ok=True, parents=True)

    # Logging to a file
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler(), logging.FileHandler(filename=f'{model_save_path}/training_logging.log')],
    )

    model = build_finetuning_model(
        base_model_name_or_path=model_name_or_path, model_args={'torch_dtype': torch.float32}
    )
    logging.info(f'pre-trained model {model_name} loaded')

    functions = datasets.load_from_disk(data_folder)
    train_functions = functions['train'].select_columns(['input_ids', 'attention_mask', 'label'])
    logging.info('training data loaded')

    train_data_sampler = LazySentenceLabelDataset(train_functions)
    train_data_loader = DataLoader(train_data_sampler, batch_size=batch_size)
    logging.info('training data created')

    def eval_triplets(functions):
        for example in tqdm(functions, desc='making InputExamples: test'):
            yield InputExample(texts=[example['anchor'], example['pos'], example['neg']])

    test_functions = functions['test']

    # The evaluator is responsible for generating the training-time metrics. We use triplets from the test-split
    # that have been formatted by the `eval_triplets` function above. The train loop (`old_fit`) will use the
    # TripletEvaluator to calculate triplet performance across this set.
    # We're going to be using old_fit to perform the actual training, and the output of TripletEvaluator is in the
    # new dictionary style. The OldFitScalarEvaluator class wraps the new-style evaluator and returns the raw metric
    # expected by `old_fit`.
    dev_evaluator = OldFitScalarEvaluator(TripletEvaluator.from_input_examples(eval_triplets(test_functions)))

    # Configure the training.
    # The jTrans loss is all triplets (including easy) with a cosine metric and a margin of 0.2 (BatchAllTripletLoss)
    # BatchSemiHardTripletLoss works on all non-easy triplets. That seems to give better training losses.
    train_loss = losses.BatchSemiHardTripletLoss(
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
    model.smart_batching_collate = collate_tokenized_batch

    # Train the model
    def train_callback(score, epoch, steps):
        print(f'{score=}, {epoch=}, {steps=}')

    # The new version of `fit` does not actually iterate the data loader for every epoch,
    # it materializes the dataset to HF format and uses the exact same data.
    # Since we rely on the loader to give us randomized batches (and therefore triplets), we fall back
    # to `old_fit` instead.
    model.old_fit(
        train_objectives=[(train_data_loader, train_loss)],
        epochs=num_epochs,
        evaluator=dev_evaluator,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_amp=use_amp,
        checkpoint_path=model_save_path,
        checkpoint_save_steps=100_000 * 16 // batch_size,
        callback=train_callback,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help='folder with input data')

    parser.add_argument('model', type=str, help='The name of the model used for finetuning')

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
