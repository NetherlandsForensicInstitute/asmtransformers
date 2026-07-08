import random
from itertools import groupby
from operator import itemgetter

from datasets import Dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm


class LazySentenceLabelDataset(IterableDataset):
    """Lazily samples tokenized examples by label for triplet-style training."""

    def __init__(self, dataset: Dataset, samples_per_label: int = 2, with_replacement: bool = False):
        super().__init__()

        self.samples_per_label = samples_per_label
        self.with_replacement = with_replacement
        self.dataset = dataset

        self.labels = dataset['label']

        # build index
        self.label2rowindices = {}
        labels_with_indices = list(enumerate(self.labels))
        labels_with_indices.sort(key=itemgetter(1))
        for label, tuples in tqdm(groupby(labels_with_indices, key=itemgetter(1)), desc='building index'):
            indices = [idx for idx, _ in tuples]
            if len(indices) >= self.samples_per_label:
                self.label2rowindices[label] = indices

    def __iter__(self):
        labels = list(self.label2rowindices.keys())
        random.shuffle(labels)

        for label in labels:
            indices = self.label2rowindices[label]

            if self.with_replacement:
                selected = random.choices(indices, k=self.samples_per_label)
            else:
                selected = random.sample(indices, k=self.samples_per_label)

            for idx in selected:
                yield self.example(idx)

    def __len__(self):
        return len(self.label2rowindices) * self.samples_per_label

    def example(self, idx):
        row = self.dataset[idx]
        return {
            'input_ids': row['input_ids'],
            'attention_mask': row['attention_mask'],
            'label': row['label'],
        }
