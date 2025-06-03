import random
from itertools import groupby
from operator import itemgetter

from datasets import Dataset
from sentence_transformers import InputExample
from torch.utils.data import IterableDataset
from tqdm import tqdm


class LazySentenceLabelDataset(IterableDataset):
    """A lazy-loading version of sentence_transformers.datasets.SentenceLabelDataset designed for use with Hugging
    Face datasets.

    This dataset class is optimized for scenarios where data is loaded from disk on-the-fly, reducing memory
    footprint. It supports sampling multiple examples per label in a lazy manner, making it suitable for training
    models with specific loss functions like triplet loss, where multiple examples of the same label are required in
    a batch.

    The dataset ensures that a specified number of samples for each label are selected either with or without
    replacement. Labels with fewer samples than required are excluded. This implementation relies on Hugging Face's
    `Dataset` format for the input data source, enabling efficient disk-based data handling.

    Parameters:
    -----------
    dataset : Dataset
        A Hugging Face `Dataset` object containing the data. Each data example must have a 'label' and other fields
        necessary for model input (e.g., text).
    samples_per_label : int, optional
        The number of samples to draw per label for each iteration. The default is 2.
    with_replacement : bool, optional
        Determines the sampling strategy. If `True`, samples are drawn with replacement, allowing the same sample
        to be selected multiple times. The default is `False`."""

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
        return sum(len(indices) for indices in self.label2rowindices.values())

    def example(self, idx):
        row = self.dataset[idx]
        return InputExample(guid=str(idx), texts=[row['cfg']], label=row['label'])
