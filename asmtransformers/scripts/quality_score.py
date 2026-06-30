import argparse
import json

import datasets
import numpy as np
import pyarrow.compute as pc


def map_tokens_to_dataset(dataset, tokenizer):
    jtp_out_of_range_token = tokenizer['model']['vocab']['JUMP_ADDR_EXCEEDED']
    jtp_unknown = tokenizer['model']['vocab']['UNK_JUMP_ADDR']
    minlength = max(jtp_out_of_range_token, jtp_unknown) + 1
    token = np.asarray(dataset['input_ids']).ravel()
    token_to_id = {t['content']: t['id'] for t in tokenizer['added_tokens']}
    padding_token = token_to_id['[PAD]']
    token = token[token != padding_token]
    bincount = np.bincount(token, minlength=minlength)

    return {
        'len_cfg': len(token),
        'jtp_in_range': int(bincount[:512].sum()),
        'jtp_out_of_range': int(bincount[jtp_out_of_range_token]),  # Jump target exceeded token (from tokenizer)
        'jtp_unknown': int(bincount[jtp_unknown]),  #  Jump Target Unknown token  (from tokenizer)
    }


SCORE_COLUMNS = ['len_cfg', 'jtp_in_range', 'jtp_unknown', 'jtp_out_of_range']


def column_stats(dataset, columns=SCORE_COLUMNS):
    """Min/max per column over a ``Dataset`` or globally across all splits of a ``DatasetDict``.

    Computing one global range keeps the normalized scores comparable across splits. Min/max are computed in C with
    pyarrow over the memory-mapped Arrow column, rather than materializing a multi-gigabyte Python list per column
    (which would thrash memory on large corpora).
    """
    splits = list(dataset.values()) if isinstance(dataset, datasets.DatasetDict) else [dataset]
    stats = {col: {'min': None, 'max': None} for col in columns}
    for split in splits:
        for col in columns:
            min_max = pc.min_max(split.data.column(col))
            split_min, split_max = min_max['min'].as_py(), min_max['max'].as_py()
            current = stats[col]
            current['min'] = split_min if current['min'] is None else min(current['min'], split_min)
            current['max'] = split_max if current['max'] is None else max(current['max'], split_max)
    return stats


def make_scorer(cfg_info):

    def normalize(val, col):
        mn, mx = cfg_info[col]['min'], cfg_info[col]['max']
        return (val - mn) / (mx - mn) if mx != mn else 0

    def add_score(example):
        """
        Here weights can be added to features for their importance if this is deemed neccesairy.
        """
        example['score'] = (
            normalize(example['len_cfg'], 'len_cfg')
            + normalize(example['jtp_in_range'], 'jtp_in_range')
            - normalize(example['jtp_out_of_range'], 'jtp_out_of_range')
            - normalize(example['jtp_unknown'], 'jtp_unknown')
        )

        return example

    return add_score


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('data-folder', type=str, required=True, help='folder with data')
    parser.add_argument('output-folder', type=str, required=True, help='folder with data')
    parser.add_argument('-t', '--tokenizer', type=str, required=True, help='path to tokenizer.json')
    return parser


def main(data_folder, tokenizer, output_folder):
    with open(tokenizer) as f:
        tokenizer = json.load(f)

    tokenized_dataset = datasets.load_from_disk(data_folder)

    # Need two passes of `map`` because to normalize we have to know min and max of values
    tokenized_dataset = tokenized_dataset.map(map_tokens_to_dataset, fn_kwargs={'tokenizer': tokenizer}, num_proc=10)

    # In this pass we determine quality_score, normalizing against a single global range across all splits
    cfg_info = column_stats(tokenized_dataset)
    scored_dataset = tokenized_dataset.map(make_scorer(cfg_info), num_proc=10)
    scored_dataset.save_to_disk(output_folder)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
