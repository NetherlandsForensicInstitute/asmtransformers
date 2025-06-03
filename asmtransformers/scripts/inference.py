import argparse

import torch
from datasets import Dataset, concatenate_datasets

from asmtransformers.models.asmsentencebert import ASMSentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def main(data_folder, output_folder, model_path):
    print("Opening dataset")
    eval_functions = Dataset.load_from_disk(data_folder)
    print('Load model')
    model = ASMSentenceTransformer.from_pretrained(model_path)
    print("Start creating embeddings")
    embedded_functions = Dataset.from_dict({'embeddings': model.encode(eval_functions['cfg'])})
    embedded_dataset = concatenate_datasets([eval_functions, embedded_functions], axis=1)
    print("Embeddings created, save to disk")
    embedded_dataset.save_to_disk(output_folder)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-d',
        '--data-folder',
        type=str,
        required=True,
        help="folder with data"
    )
    parser.add_argument(
        '-o',
        '--output-folder',
        type=str,
        required=True,
        help="folder with data"
    )

    parser.add_argument(
        '-m',
        '--model-path',
        type=str,
        required=True,
        help="model"
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
