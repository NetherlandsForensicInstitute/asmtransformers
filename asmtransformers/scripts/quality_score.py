import json
import sys
from pathlib import Path

import datasets
import numpy as np


OUTPUT = Path('./results')


def map_tokens_to_dataset(dataset, tokenizer):
    jtp_out_of_range_token = tokenizer['model']['vocab']['JUMP_ADDR_EXCEEDED']
    jtp_unknown = tokenizer['model']['vocab']['UNK_JUMP_ADDR']
    minlength = max(jtp_out_of_range_token, jtp_unknown) + 1
    token = np.asarray(dataset['input_ids']).ravel()
    token_to_id = {t['content']: t['id'] for t in tokenizer['added_tokens']}
    padding_token = token_to_id['[SEP]']
    token = token[token != padding_token]
    bincount = np.bincount(token, minlength=minlength)

    return {
        'len_cfg': len(token),
        'jtp_in_range': int(bincount[:512].sum()),
        'jtp_out_of_range': int(bincount[jtp_out_of_range_token]),  # Jump target exceeded token (from tokenizer)
        'jtp_unknown': int(bincount[jtp_unknown]),  #  Jump Target Unknown token  (from tokenizer)
    }


def make_scorer(dataset):

    cfg_info = {
        col: {'min': min(dataset[col]), 'max': max(dataset[col])}
        for col in ['len_cfg', 'jtp_in_range', 'jtp_unknown', 'jtp_out_of_range']
    }

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


if __name__ == '__main__':
    # Make sure tokenized dataset has been made With mktokenizer and tokenize_dataset.py
    dataset_path, tokenizer_path = sys.argv[1:]
    with open(tokenizer_path) as f:
        tokenizer = json.load(f)

    tokenized_dataset = datasets.load_from_disk(dataset_path)

    # Need two passes of `map`` because to normalize we have to know min and max of values
    tokenized_dataset = tokenized_dataset.map(map_tokens_to_dataset, fn_kwargs={'tokenizer': tokenizer}, num_proc=10)

    # In this pass we determine quality_score
    scored_dataset = tokenized_dataset.map(make_scorer(tokenized_dataset), num_proc=10)
    scored_dataset.save_to_disk(OUTPUT)
