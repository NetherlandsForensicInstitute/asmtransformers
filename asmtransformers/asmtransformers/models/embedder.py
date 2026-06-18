from itertools import batched

import numpy as np
import torch

from asmtransformers.models.asmbert import ASMBertModel, ASMTokenizer
from asmtransformers.models.finetuning import mean_pool_embeddings


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
        return mean_pool_embeddings(token_embeddings, attention_mask)

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
