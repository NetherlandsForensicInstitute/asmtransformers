import json
import sys

from datasets import Dataset
from transformers import BertTokenizer

from asmtransformers import arm64, operands


def preprocess(tokenizer, dataset):
    preprocessor = arm64.Preprocessor(operand_formatters=(
        # 2-log numerical values and offsets to reduce the number of unique tokens we'll generate
        # (pre-made vocabulary used this too)
        operands.format_immediate_log,
        operands.format_offset_log,
    ))

    def tokenize(function):
        # control flow graph of a function is saved as a list of 2-tuples, preprocessor expects a dict
        cfg = dict(json.loads(function['cfg']))
        # apply the bert tokenizer on the preprocessed version of the function's control flow graph
        return tokenizer(
            preprocessor.preprocess(cfg),
            # input is pre-tokenized by the preprocessor
            is_split_into_words=True,
            # TODO: is this needed (which special tokens do we need it to add)?
            #       (as the tokenizer is pre-configured with a vocab, we might at least need [UNK]?)
            add_special_tokens=True,
            # truncate the encoded sequence to a maximum of max_length tokens
            truncation=True,
            max_length=512,
            # TODO: ???
            return_special_tokens_mask=True,
        )

    return dataset.map(tokenize, batched=False, num_proc=10)


if __name__ == '__main__':
    # expect 3 arguments from cli
    tokenizer, data_in, data_out = sys.argv[1:]

    # let the tokenizer preprocess data from data_in, write the result to data_out
    preprocess(
        BertTokenizer.from_pretrained(tokenizer),
        Dataset.load_from_disk(data_in),
    ).save_to_disk(data_out)
