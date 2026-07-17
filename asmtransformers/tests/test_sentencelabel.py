from datasets import Dataset

from asmtransformers.datasets import LazySentenceLabelDataset


def test_len_matches_number_of_yielded_examples():
    source = Dataset.from_dict(
        {
            'input_ids': [[1], [2], [3], [4], [5], [6], [7]],
            'attention_mask': [[1], [1], [1], [1], [1], [1], [1]],
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
    assert [example['label'] for example in yielded].count('a') == 2
    assert [example['label'] for example in yielded].count('b') == 2
    assert [example['label'] for example in yielded].count('c') == 0
    assert all(set(example) == {'input_ids', 'attention_mask', 'label'} for example in yielded)
