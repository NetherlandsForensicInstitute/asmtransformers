from datasets import Dataset

from asmtransformers.datasets import LazySentenceLabelDataset


def test_len_matches_number_of_yielded_examples():
    source = Dataset.from_dict({
        'cfg': [
            'a-0', 'a-1',
            'b-0', 'b-1', 'b-2', 'b-3',
            'c-0',
        ],
        'label': [
            'a', 'a',
            'b', 'b', 'b', 'b',
            'c',
        ],
    })

    dataset = LazySentenceLabelDataset(source, samples_per_label=2)

    yielded = list(dataset)

    assert len(dataset) == 4
    assert len(yielded) == len(dataset)
    assert [example.label for example in yielded].count('a') == 2
    assert [example.label for example in yielded].count('b') == 2
    assert [example.label for example in yielded].count('c') == 0
