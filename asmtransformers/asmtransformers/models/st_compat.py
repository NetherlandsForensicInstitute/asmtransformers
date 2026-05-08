import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling

from .asmbert import ASMTokenizer
from .asmsentencebert import ASMTransformerModule, build_finetuning_model
from .embedder import ASMEmbedder


class STCheckpointTransformerModule(ASMTransformerModule):
    """Compatibility module for existing sentence-transformers checkpoints.

    Remove this when published embedding checkpoints are converted away from the
    old sentence-transformers module format.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_seq_length: int | None = None,
        do_lower_case: bool = False,
        model_args: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_name_or_path, model_args=model_args)
        if processor_kwargs:
            self.tokenizer = ASMTokenizer.from_pretrained(model_name_or_path, **processor_kwargs)
            self.model.tokenizer = self.tokenizer
        self.max_seq_length = max_seq_length or min(
            self.model.config.max_position_embeddings,
            self.tokenizer.model_max_length,
        )
        self.do_lower_case = do_lower_case

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        output_path = Path(output_path)
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)
        (output_path / 'sentence_bert_config.json').write_text(
            json.dumps(
                {
                    'max_seq_length': self.max_seq_length,
                    'do_lower_case': self.do_lower_case,
                }
            )
        )

    @classmethod
    def load(cls, model_name_or_path: str, model_kwargs=None, processor_kwargs=None, **kwargs):
        config_path = Path(model_name_or_path) / 'sentence_bert_config.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return cls(
            model_name_or_path,
            max_seq_length=config.get('max_seq_length'),
            do_lower_case=config.get('do_lower_case', False),
            model_args=model_kwargs or {},
            processor_kwargs=processor_kwargs or {},
        )


# Older sentence-transformers checkpoints reference this exact symbol in modules.json.
ASMSTTransformer = STCheckpointTransformerModule


def build_sentence_transformer(model_name_or_path, model_args=None):
    embedding_model = STCheckpointTransformerModule(model_name_or_path, model_args=model_args or {})
    pooling_model = Pooling(embedding_model.get_embedding_dimension())
    return _normalize_encode_by_default(SentenceTransformer(modules=[embedding_model, pooling_model]))


def _normalize_encode_by_default(model):
    encode = model.encode

    def encode_with_normalization_default(sentences, *args, normalize_embeddings=True, **kwargs):
        return encode(sentences, *args, normalize_embeddings=normalize_embeddings, **kwargs)

    model.encode = encode_with_normalization_default
    return model


class ASMSentenceTransformer:
    """Compatibility factory for existing callers.

    New code should use build_finetuning_model().
    Remove this when old caller imports are migrated.
    """

    from_pretrained = staticmethod(build_sentence_transformer)
    from_basemodel = staticmethod(build_finetuning_model)


def load_st_embedding_as_native_embedder(model_name_or_path, *, model_args=None, tokenizer_args=None, device=None):
    """Load an old ST-format embedding checkpoint through the native embedder.

    Remove this when published embedding checkpoints are converted to native
    ASMBertModel/ASMTokenizer checkpoint directories.
    """

    model_path = _resolve_transformer_path(model_name_or_path)
    _validate_pooling(model_name_or_path)
    return ASMEmbedder.from_pretrained(
        model_path,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        device=device,
    )


def _resolve_transformer_path(model_name_or_path):
    path = Path(model_name_or_path)
    modules_path = path / 'modules.json'
    if not modules_path.exists():
        return model_name_or_path

    modules = json.loads(modules_path.read_text())
    transformer = next((module for module in modules if module.get('idx') == 0), None)
    if transformer is None:
        return model_name_or_path

    transformer_path = transformer.get('path') or ''
    return str(path / transformer_path)


def _validate_pooling(model_name_or_path):
    path = Path(model_name_or_path)
    modules_path = path / 'modules.json'
    if not modules_path.exists():
        return

    modules = json.loads(modules_path.read_text())
    pooling = next((module for module in modules if 'Pooling' in module.get('type', '')), None)
    if pooling is None:
        return

    config_path = path / pooling.get('path', '') / 'config.json'
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    if config.get('pooling_mode', 'mean') != 'mean':
        raise ValueError(f'Unsupported pooling mode: {config["pooling_mode"]}')
