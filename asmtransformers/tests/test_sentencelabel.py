from datasets import Dataset

from asmtransformers.datasets import LazySentenceLabelDataset, as_sentence_transformer_training_dataset


def test_training_dataset_renames_cfg_to_sentence():
    source = Dataset.from_dict(
        {
            'cfg': ['a-0', 'a-1'],
            'label': [0, 1],
            'ignored': ['x', 'y'],
        }
    )

    dataset = as_sentence_transformer_training_dataset(source)

    assert dataset.column_names == ['sentence', 'label']
    assert dataset['sentence'] == ['a-0', 'a-1']
    assert dataset['label'] == [0, 1]


def test_len_matches_number_of_yielded_examples():
    source = Dataset.from_dict(
        {
            'cfg': [
                'a-0',
                'a-1',
                'b-0',
                'b-1',
                'b-2',
                'b-3',
                'c-0',
            ],
            'label': [
                'a',
                'a',
                'b',
                'b',
                'b',
                'b',
                'c',
            ],
        }
    )

    dataset = LazySentenceLabelDataset(source, samples_per_label=2)

    yielded = list(dataset)

    assert len(dataset) == 4
    assert len(yielded) == len(dataset)
    assert [example.label for example in yielded].count('a') == 2
    assert [example.label for example in yielded].count('b') == 2
    assert [example.label for example in yielded].count('c') == 0
