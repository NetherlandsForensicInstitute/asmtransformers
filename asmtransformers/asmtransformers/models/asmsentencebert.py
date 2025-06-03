from collections.abc import Mapping
from typing import Any

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from torch import nn
from transformers import BertTokenizer

from .asmbert import ARM64Tokenizer, ASMBertModel


class ASMSTTransformer(Transformer):
    """Analogous to the sentence-transformers Transformer class,
    managing our code transformers and tokenizer.

    See ASMSentenceTransformer for an overall description."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer,
        max_seq_length: int | None = None,
        do_lower_case: bool = False,
        model_args: Mapping[str, Any] | None = None,
    ):
        nn.Module.__init__(self)

        if model_args is None:
            model_args = {}

        model = ASMBertModel.from_pretrained(model_name_or_path, **model_args)
        model.tokenizer = tokenizer
        self.tokenizer = tokenizer

        self.auto_model = model

        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        # No max_seq_length set. Try to infer from model
        if (
            max_seq_length is None
            and hasattr(self.auto_model, 'config')
            and hasattr(self.auto_model.config, 'max_position_embeddings')
            and hasattr(self.tokenizer, 'model_max_length')
        ):
            max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


class ASMSentenceTransformer(SentenceTransformer):
    """Convenience class that allows for easy finetuning and inference for a
    semantic code similarity model.

    Exposes the same interface as sentence-transformer's class SentenceTransformer
    does; particularly the `fit()` method for finetuning and the `encode()` method
    for inference.

    SentenceTransformer internally composes model and tokenizer classes from the
    Hugging Face transformer library, so we do the same for our asmtransformers
    models and tokenizer.

    Graphically:
    ("ST.": comes from the sentence-transformers package
     "T.": comes from the HuggingFace transformers package)

        ASMSentenceTransformer                 ST.SentenceTransformer
               |                                      |
          ----------------------                    -------------------
          |                    |                    |                 |
        ASMSTTransformer   ST.Pooling           ST.Transformer    ST.Pooling
         |                                         |
         --------------------                     -----------------
        |                   |                     |               |
     ASMBertModel    ARM64BertTokenizer      T.BertModel    T.BertTokenizer
    """

    def __init__(self, embedding_model, pooling_model, tokenizer):
        super().__init__(modules=[embedding_model, pooling_model])
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        tokenizer = ARM64Tokenizer.from_pretrained(model_name_or_path)
        tokenizer.set_tokenizer(BertTokenizer.from_pretrained(model_name_or_path))

        embedding_model = ASMSTTransformer(model_name_or_path, tokenizer, model_args or {})
        pooling_model = Pooling(embedding_model.get_word_embedding_dimension())
        return cls(embedding_model, pooling_model, tokenizer)

    @classmethod
    def from_basemodel(cls, base_model_name_or_path, model_args=None):
        # Load the sentenceTransformer model. This line implicitly creates a sentence transformer with a new pooling
        # layer and warns about it.
        embedding_model = Transformer(model_name_or_path=base_model_name_or_path, model_args=model_args or {})

        # overwrite the underlying BERT model with our own ASMBert Model
        # basically we are loading a full BERT model first and discarding(?) it, which may have memory implications
        embedding_model.auto_model = ASMBertModel.from_pretrained(base_model_name_or_path)
        tokenizer = ARM64Tokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.set_tokenizer(BertTokenizer.from_pretrained(base_model_name_or_path))

        # The jTrans architecture shares weights between positional and word embeddings
        # Make sure we have done this properly.
        if (
            embedding_model.auto_model.base_model.embeddings.position_embeddings
            is not embedding_model.auto_model.base_model.embeddings.word_embeddings
        ):
            raise RuntimeError('Word embeddings and position embeddings not shared')

        # Now freeze layers, like jTrans. Embedding plus 10 layers is the default, so we'll use that.
        for param in embedding_model.auto_model.base_model.embeddings.parameters():
            param.requires_grad = False

        freeze_layer_count = 10
        if freeze_layer_count:
            for layer in embedding_model.auto_model.base_model.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

        pooling_model = Pooling(embedding_model.get_word_embedding_dimension())

        return cls(embedding_model, pooling_model, tokenizer)

    def encode(self, sentences, *args, normalize_embeddings=True, **kwargs):
        # Change the default for normalize_embeddings.
        return super().encode(sentences, *args, normalize_embeddings=normalize_embeddings, **kwargs)
