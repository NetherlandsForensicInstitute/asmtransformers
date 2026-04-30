from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import nn

from .asmbert import ASMBertModel, ASMTokenizer


class ASMTransformerModule(nn.Module):
    """Minimal sentence-transformers module for ARM64BERT finetuning."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        model_args: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model = ASMBertModel.from_pretrained(model_name_or_path, **(model_args or {}))
        self.tokenizer = ASMTokenizer.from_pretrained(model_name_or_path)

        self.model.tokenizer = self.tokenizer
        self.model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def preprocess(self, inputs, prompt=None, **kwargs):
        if prompt:
            inputs = [prompt + text for text in inputs]
        return self.tokenizer(inputs, **kwargs)

    def tokenize(self, texts, **kwargs):
        return self.preprocess(texts, **kwargs)

    def forward(self, features: dict[str, Any], **kwargs) -> dict[str, Any]:
        model_inputs = {
            'input_ids': features['input_ids'],
            'attention_mask': features['attention_mask'],
            'return_dict': True,
        }
        if 'token_type_ids' in features:
            model_inputs['token_type_ids'] = features['token_type_ids']

        outputs = self.model(**model_inputs)
        features['token_embeddings'] = outputs.last_hidden_state
        return features


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
):
    embedding_model = ASMTransformerModule(base_model_name_or_path, model_args=model_args)
    pooling_model = Pooling(embedding_model.get_embedding_dimension())
    model = SentenceTransformer(modules=[embedding_model, pooling_model])
    bert_model = model[0].model.base_model

    if bert_model.embeddings.position_embeddings is not bert_model.embeddings.word_embeddings:
        raise RuntimeError('Word embeddings and position embeddings not shared')

    apply_freeze_policy(
        bert_model,
        freeze_embeddings=freeze_embeddings,
        freeze_layer_count=freeze_layer_count,
    )
    return model


def __getattr__(name):
    if name in {'ASMSentenceTransformer', 'ASMSTTransformer'}:
        # Temporary checkpoint compatibility bridge: published ST-format checkpoints
        # reference asmtransformers.models.asmsentencebert.ASMSTTransformer in modules.json.
        # Delete this after those checkpoints are converted to native embedder format.
        from .st_compat import ASMSentenceTransformer, ASMSTTransformer

        return {'ASMSentenceTransformer': ASMSentenceTransformer, 'ASMSTTransformer': ASMSTTransformer}[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
