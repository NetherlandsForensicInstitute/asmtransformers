from collections.abc import Mapping
from typing import Any

import torch

from .asmbert import ASMBertModel, ASMTokenizer


def mean_pool_embeddings(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


class ASMFinetuningModel(torch.nn.Module):
    """Native embedding model used for triplet-loss finetuning."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        model_args: Mapping[str, Any] | None = None,
        tokenizer_args: Mapping[str, Any] | None = None,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.model = ASMBertModel.from_pretrained(model_name_or_path, **(model_args or {}))
        self.tokenizer = ASMTokenizer.from_pretrained(model_name_or_path, **(tokenizer_args or {}))
        self.normalize_embeddings = normalize_embeddings

        self.model.tokenizer = self.tokenizer
        self.model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_dict': True,
        }
        if token_type_ids is not None:
            model_inputs['token_type_ids'] = token_type_ids

        outputs = self.model(**model_inputs)
        embeddings = mean_pool_embeddings(outputs.last_hidden_state, attention_mask)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode_batch(self, cfgs, *, architecture='arm64', device=None) -> torch.Tensor:
        inputs = self.tokenizer(cfgs, architecture=architecture)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        return self(**inputs)

    def save_pretrained(self, output_path: str, *, safe_serialization: bool = True) -> None:
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)


def apply_freeze_policy(bert_model, *, freeze_embeddings=True, freeze_layer_count=10):
    if freeze_embeddings:
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False

    if freeze_layer_count:
        for layer in bert_model.encoder.layer[:freeze_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False


def build_finetuning_model(
    base_model_name_or_path,
    model_args=None,
    *,
    freeze_embeddings=True,
    freeze_layer_count=10,
    normalize_embeddings=True,
):
    model = ASMFinetuningModel(
        base_model_name_or_path,
        model_args=model_args,
        normalize_embeddings=normalize_embeddings,
    )
    bert_model = model.model.base_model

    if bert_model.embeddings.position_embeddings is not bert_model.embeddings.word_embeddings:
        raise RuntimeError('Word embeddings and position embeddings not shared')

    apply_freeze_policy(
        bert_model,
        freeze_embeddings=freeze_embeddings,
        freeze_layer_count=freeze_layer_count,
    )
    return model


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return 1 - normalized @ normalized.T


def batch_semi_hard_triplet_loss(
    labels: torch.Tensor, embeddings: torch.Tensor, *, margin: float = 0.2
) -> torch.Tensor:
    if torch.unique(labels).numel() < 2:
        return embeddings.sum() * 0

    labels = labels.unsqueeze(1)
    pairwise_distances = cosine_distance_matrix(embeddings)

    adjacency = labels == labels.T
    adjacency_not = ~adjacency

    batch_size = torch.numel(labels)
    distances_tile = pairwise_distances.repeat(batch_size, 1)
    anchor_positive_distances = pairwise_distances.T.reshape(-1, 1)
    mask = adjacency_not.repeat(batch_size, 1) & (distances_tile > anchor_positive_distances)

    mask_final = torch.sum(mask, dim=1, keepdim=True) > 0
    mask_final = mask_final.reshape(batch_size, batch_size).T

    negatives_outside = _masked_minimum(distances_tile, mask).reshape(batch_size, batch_size).T
    negatives_inside = _masked_maximum(pairwise_distances, adjacency_not).repeat(1, batch_size)
    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_matrix = pairwise_distances - semi_hard_negatives + margin
    positive_mask = adjacency.float().to(labels.device) - torch.eye(batch_size, device=labels.device)
    positive_count = torch.sum(positive_mask)

    if positive_count == 0:
        return embeddings.sum() * 0

    return torch.sum(torch.clamp(loss_matrix * positive_mask, min=0)) / positive_count


def _masked_minimum(data: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    axis_maximums, _ = data.max(dim, keepdim=True)
    masked_minimums = (data - axis_maximums) * mask
    masked_minimums, _ = masked_minimums.min(dim, keepdim=True)
    return masked_minimums + axis_maximums


def _masked_maximum(data: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    axis_minimums, _ = data.min(dim, keepdim=True)
    masked_maximums = (data - axis_minimums) * mask
    masked_maximums, _ = masked_maximums.max(dim, keepdim=True)
    return masked_maximums + axis_minimums
