from itertools import batched

import numpy as np
import torch
from tqdm import tqdm

from asmtransformers.models.asmbert import ASMBertModel, ASMTokenizer


class ASMEmbedder:
    """Native inference wrapper for ARM64BERT-style embedding checkpoints."""

    def __init__(self, model, tokenizer, *, device=None, normalize_embeddings=True):
        self.model = model
        self.tokenizer = tokenizer
        self.normalize_embeddings = normalize_embeddings
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls, model_name_or_path, *, model_args=None, tokenizer_args=None, device=None, normalize_embeddings=True
    ):
        tokenizer = ASMTokenizer.from_pretrained(model_name_or_path, **(tokenizer_args or {}))
        model = ASMBertModel.from_pretrained(model_name_or_path, **(model_args or {}))
        return cls(model, tokenizer, device=device, normalize_embeddings=normalize_embeddings)

    @staticmethod
    def mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

    def encode(
        self,
        sentences,
        *,
        batch_size=32,
        architecture='arm64',
        normalize_embeddings=None,
        convert_to_numpy=True,
    ):
        single_input = isinstance(sentences, str)
        sentences = [sentences] if single_input else sentences
        normalize_embeddings = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        embeddings = []
        with torch.no_grad():
            for batch in batched(sentences, batch_size, strict=False):
                inputs = self.tokenizer(batch, architecture=architecture)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                pooled = self.mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        if single_input:
            embeddings = embeddings[0]
        if convert_to_numpy:
            return embeddings.numpy().astype(np.float32, copy=False)
        return embeddings

    def turn_into_tensors(self, batch):
        # there might be a smarter way to do this, suggestions welcome!
        features = {
            'input_ids': torch.tensor([example[0] for example in batch], dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor([example[1] for example in batch], dtype=torch.long).to(self.device),
        }
        return features

    def get_embeddings(
        self,
        token_ids,
        attention_mask,
        *,
        batch_size=32,
        normalize_embeddings=None,
        convert_to_numpy=True,
    ):
        normalize_embeddings = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        inputs = zip(token_ids, attention_mask, strict=True)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(batched(inputs, batch_size, strict=False)):
                inputs = self.turn_into_tensors(batch, self.device)
                outputs = self.model(**inputs)
                pooled = self.mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        if convert_to_numpy:
            return embeddings.numpy().astype(np.float32, copy=False)
        return embeddings
