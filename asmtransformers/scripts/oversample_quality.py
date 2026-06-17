import argparse
import logging

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


def oversample_indices(scores, *, top_fraction, repeats):
    """Return row indices that keep every function once and repeat the top-scoring ones ``repeats`` extra times.

    ``top_fraction`` is the fraction of the corpus treated as high quality (e.g. 0.25 for the top 25%). The threshold
    is the corresponding score quantile; ties on the boundary are all kept, so the high-quality slice can be slightly
    larger than ``top_fraction``.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f'top_fraction must be in (0, 1], got {top_fraction}')
    if repeats < 0:
        raise ValueError(f'repeats must be >= 0, got {repeats}')

    scores = np.asarray(scores, dtype=np.float64)
    threshold = np.quantile(scores, 1 - top_fraction)
    high = np.flatnonzero(scores >= threshold)

    base = np.arange(len(scores))
    extra = np.tile(high, repeats)
    return np.concatenate([base, extra]), threshold, len(high)


def oversample_train(train, *, score_column, top_fraction, repeats, seed):
    if score_column not in train.column_names:
        raise KeyError(
            f'score column {score_column!r} not found; run scripts/quality_score.py first '
            f'(available columns: {train.column_names})'
        )

    indices, threshold, num_high = oversample_indices(train[score_column], top_fraction=top_fraction, repeats=repeats)
    logging.info(
        'Oversampling: %d functions, %d above threshold %.4f, each repeated %d extra time(s) -> %d training rows',
        len(train),
        num_high,
        threshold,
        repeats,
        len(indices),
    )
    return train.select(indices).shuffle(seed=seed)


def build_oversampled_dataset(scored, *, score_column, top_fraction, repeats, seed):
    """Oversample the high-quality functions and return a pretrain-ready ``DatasetDict`` with a ``train`` split."""
    if isinstance(scored, DatasetDict):
        if 'train' not in scored:
            raise KeyError(f"expected a 'train' split, got splits {list(scored)}")
        oversampled = DatasetDict(scored)
        oversampled['train'] = oversample_train(
            scored['train'], score_column=score_column, top_fraction=top_fraction, repeats=repeats, seed=seed
        )
        if 'test' not in oversampled:
            logging.warning("no 'test' split present; intermediate evaluation will be disabled during pretraining")
        return oversampled

    if isinstance(scored, Dataset):
        logging.warning("input is a bare Dataset; wrapping the oversampled result as DatasetDict({'train': ...})")
        return DatasetDict(
            {
                'train': oversample_train(
                    scored, score_column=score_column, top_fraction=top_fraction, repeats=repeats, seed=seed
                )
            }
        )

    raise TypeError(f'unsupported dataset type {type(scored).__name__}')


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Build a quality-oversampled training set for the final pretraining (annealing) phase'
    )
    parser.add_argument('input', help='path to the scored dataset saved by scripts/quality_score.py')
    parser.add_argument('output', help='path to write the oversampled DatasetDict (save_to_disk)')
    parser.add_argument(
        '--top-fraction',
        type=float,
        default=0.25,
        help='fraction of the corpus treated as high quality (default: top 25%%)',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='number of extra copies of each high-quality function (default: 3)',
    )
    parser.add_argument('--score-column', default='score', help="name of the quality-score column (default: 'score')")
    parser.add_argument('--seed', type=int, default=42, help='shuffle seed for the oversampled training rows')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    args = build_arg_parser().parse_args()

    scored = load_from_disk(args.input)
    oversampled = build_oversampled_dataset(
        scored,
        score_column=args.score_column,
        top_fraction=args.top_fraction,
        repeats=args.repeats,
        seed=args.seed,
    )
    oversampled.save_to_disk(args.output)
    logging.info('Saved oversampled dataset to %s', args.output)
