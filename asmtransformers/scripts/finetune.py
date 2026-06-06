import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from tzlocal import get_localzone

from asmtransformers.datasets import LazySentenceLabelDataset
from asmtransformers.models.finetuning import batch_semi_hard_triplet_loss, build_finetuning_model


def timestamp():
    return dt.datetime.now(tz=get_localzone()).strftime('%Y-%m-%d_%H-%M-%S')


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def collate_labeled_batch(tokenizer, examples, *, architecture='arm64'):
    cfgs = [example['cfg'] for example in examples]
    labels = torch.tensor([example['label'] for example in examples], dtype=torch.long)
    batch = tokenizer(cfgs, architecture=architecture)
    batch['labels'] = labels
    return batch


def encode_cfgs(model, cfgs, *, batch_size=32, architecture='arm64', device=None):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(cfgs), batch_size):
            inputs = model.tokenizer(cfgs[offset : offset + batch_size], architecture=architecture)
            if device is not None:
                inputs = move_to_device(inputs, device)
            embeddings.append(model(**inputs).cpu())
    return torch.cat(embeddings, dim=0)


def evaluate_triplets(model, triplets, output_path, *, batch_size=32, architecture='arm64', epoch=-1, steps=-1):
    anchors = [example['anchor'] for example in triplets]
    positives = [example['pos'] for example in triplets]
    negatives = [example['neg'] for example in triplets]
    device = next(model.parameters()).device

    anchor_embeddings = encode_cfgs(model, anchors, batch_size=batch_size, architecture=architecture, device=device)
    positive_embeddings = encode_cfgs(model, positives, batch_size=batch_size, architecture=architecture, device=device)
    negative_embeddings = encode_cfgs(model, negatives, batch_size=batch_size, architecture=architecture, device=device)

    positive_scores = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
    negative_scores = torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings)
    accuracy = (positive_scores > negative_scores).float().mean().item()

    eval_path = Path(output_path) / 'eval'
    eval_path.mkdir(exist_ok=True)
    csv_path = eval_path / 'triplet_evaluation_results.csv'
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(['epoch', 'steps', 'cosine_accuracy'])
        writer.writerow([epoch, steps, accuracy])

    logging.info(f'Triplet cosine accuracy: {accuracy:.2%}')
    return accuracy


def save_checkpoint(model, model_save_path, step):
    checkpoint_path = Path(model_save_path) / f'checkpoint-{step}'
    checkpoint_path.mkdir(exist_ok=True)
    model.save_pretrained(str(checkpoint_path))


def main(data_folder, model, batch_size):
    """
    This script takes a language model and finetunes it for semantic similarity.

    :param data_folder: a folder with a saved Hugging Face DatasetDict
    :param model: the model to be finetuned
    :param batch_size: the batch size
    """

    model_name_or_path = model
    model_name = Path(model_name_or_path).stem if Path(model_name_or_path).is_dir() else model_name_or_path
    num_epochs = 3
    evaluation_steps = 50_000
    warmup_steps = 500
    learning_rate = 2e-5

    model_save_path = f'output/aarch64_ft_{model_name.replace("/", "-")}-{timestamp()}'
    Path(model_save_path).mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=f'{model_save_path}/training_logging.log'),
        ],
    )

    finetuning_model = build_finetuning_model(
        base_model_name_or_path=model_name_or_path, model_args={'torch_dtype': torch.float32}
    )
    logging.info(f'pre-trained model {model_name} loaded')

    functions = datasets.load_from_disk(data_folder)
    train_functions = functions['train']
    test_functions = functions['test']
    logging.info('training data loaded')

    train_data_sampler = LazySentenceLabelDataset(train_functions)
    train_data_loader = DataLoader(
        train_data_sampler,
        batch_size=batch_size,
        collate_fn=lambda examples: collate_labeled_batch(finetuning_model.tokenizer, examples),
    )
    logging.info('training data created')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finetuning_model.to(device)

    optimizer = torch.optim.AdamW(
        [param for param in finetuning_model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    total_steps = len(train_data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    global_step = 0
    checkpoint_save_steps = 100_000 * 16 // batch_size

    for epoch in range(num_epochs):
        finetuning_model.train()
        progress = tqdm(train_data_loader, desc=f'epoch {epoch + 1}/{num_epochs}')
        for batch in progress:
            labels = batch.pop('labels').to(device)
            inputs = move_to_device(batch, device)

            optimizer.zero_grad()
            embeddings = finetuning_model(**inputs)
            loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = loss.item()
            progress.set_postfix(loss=loss_value)
            with open(Path(model_save_path) / 'train_loss.log', 'a') as loss_log:
                loss_log.write(f'{loss_value}\n')

            if global_step % evaluation_steps == 0:
                score = evaluate_triplets(
                    finetuning_model,
                    test_functions,
                    model_save_path,
                    batch_size=batch_size,
                    epoch=epoch,
                    steps=global_step,
                )
                print(f'score={score}, epoch={epoch}, steps={global_step}')
                finetuning_model.train()

            if global_step % checkpoint_save_steps == 0:
                save_checkpoint(finetuning_model, model_save_path, global_step)

    evaluate_triplets(
        finetuning_model,
        test_functions,
        model_save_path,
        batch_size=batch_size,
        epoch=num_epochs - 1,
        steps=global_step,
    )
    finetuning_model.save_pretrained(model_save_path)


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
