import numpy as np
import torch

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
    def from_pretrained(cls, model_name_or_path, *, model_args=None, tokenizer_args=None, device=None):
        tokenizer = ASMTokenizer.from_pretrained(model_name_or_path, **(tokenizer_args or {}))
        model = ASMBertModel.from_pretrained(model_name_or_path, **(model_args or {}))
        return cls(model, tokenizer, device=device)

    @staticmethod
    def mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

    def encode(
        self,
        sentences,
        *,
        batch_size=32,
        normalize_embeddings=None,
        convert_to_numpy=True,
    ):
        single_input = isinstance(sentences, str)
        sentences = [sentences] if single_input else list(sentences)
        normalize_embeddings = self.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        embeddings = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start : start + batch_size]
                inputs = self.tokenizer(batch)
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
