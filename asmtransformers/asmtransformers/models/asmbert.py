import json
import warnings
from copy import deepcopy

import torch
from transformers import BertForMaskedLM, BertModel, BertTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel

from asmtransformers import operands
from asmtransformers.arm64 import Preprocessor


class ASMBertModel(BertModel):
    """Finetuning / inference model for _jTrans: Jump-Aware Transformer for Binary Code Similarity_.

    Based on the descriptions in https://arxiv.org/abs/2205.12713 and https://github.com/vul337/jTrans (where the
    model is referred to as BinBert).

    BinBert shares parameters between the position embeddings and the jump target embeddings.
    """

    _tied_weights_keys = {
        **(BertModel._tied_weights_keys or {}),
        'embeddings.position_embeddings.weight': 'embeddings.word_embeddings.weight',
    }

    def __init__(self, config, add_pooling_layer=True, *, delay_tie_for_load=False):
        super().__init__(config, add_pooling_layer)

        if delay_tie_for_load:
            # Keep a checkpoint-shaped positional table around just for loading
            # legacy checkpoints that only store the shared embedding under the
            # position-embedding key.
            self.embeddings.position_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.tie_shared_embeddings()

    def _tie_weights(self):
        """Declare and re-apply the custom embedding alias for HF save/load flows."""

        self.tie_shared_embeddings()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        output_loading_info = kwargs.pop('output_loading_info', False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=True,
            delay_tie_for_load=True,
            **kwargs,
        )

        if 'embeddings.word_embeddings.weight' in loading_info['missing_keys']:
            warnings.warn(
                (
                    f'Loaded legacy {cls.__name__} checkpoint from {pretrained_model_name_or_path!r} '
                    'that stores the shared embedding only under '
                    '`embeddings.position_embeddings.weight`. This compatibility '
                    'path will be removed in a future release; re-save the model '
                    'with the current asmtransformers version.'
                ),
                UserWarning,
                stacklevel=2,
            )
            model.embeddings.word_embeddings.weight.data.copy_(model.embeddings.position_embeddings.weight.data)
            loading_info['missing_keys'].discard('embeddings.word_embeddings.weight')
        model.tie_shared_embeddings()

        if output_loading_info:
            return model, loading_info
        return model

    def tie_shared_embeddings(self):
        # share parameters between position embeddings and jump target embeddings
        # (the first 512 tokens in the vocab)
        # https://github.com/vul337/jTrans/issues/3#issuecomment-1661876440
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


class ASMBertForMaskedLM(BertForMaskedLM):
    """Pre-training model for _jTrans: Jump-Aware Transformer for Binary Code Similarity_.

    Based on the descriptions in https://arxiv.org/abs/2205.12713 and https://github.com/vul337/jTrans (where the
    model is referred to as BinBert). Adds an extra pre-training task called _Jump Target Prediction_ to a standard
    BERT with a Masked Language Modelling pre-training task.

    Additionally, BinBert shares parameters between the position embeddings and the jump target embeddings.
    """

    _tied_weights_keys = {
        **BertForMaskedLM._tied_weights_keys,
        'bert.embeddings.position_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    }

    def __init__(self, config):
        """Override BertForMaskedLM's init to add Jump Target Prediction

        Because BertForMaskedLM's init ends with a call to self.post_init(), we don't want to do that full init and
        only add our JTP-related code afterwards. Therefore, we choose to completely override BertForMaskedLM's init,
        and we have to manually take care to initialize everything that needs to be initialized there."""

        # init BertForMaskedLM's superclass
        BertPreTrainedModel.__init__(self, config)

        # Initialize standard BERT model and MLM classifier as in BertForMaskedLM
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # add Jump Target Prediction classifier
        self.config_jtp = deepcopy(config)
        self.config_jtp.vocab_size = config.max_position_embeddings  # only the jump targets are possible predictions
        self.cls_jtp = BertOnlyMLMHead(self.config_jtp)

        # Initialize weights and apply final processing
        self.post_init()
        self._tie_weights()

    def _tie_weights(self):
        """Declare and re-apply the custom embedding alias for HF save/load flows."""

        # share parameters between position embeddings and jump target embeddings
        # (the first 512 tokens in the vocab)
        # https://github.com/vul337/jTrans/issues/3#issuecomment-1661876440
        self.base_model.embeddings.position_embeddings = self.base_model.embeddings.word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> MaskedLMOutput:
        r"""Overwrite BertForMaskedLM's forward method to implement Jump Target Prediction in addition to Masked
        Language Modelling.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token

        # Masked Language Modelling
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = loss_fct(input=prediction_scores.view(-1, self.config.vocab_size), target=labels.view(-1))
        masked_lm_loss = torch.nan_to_num(masked_lm_loss)

        # Jump Target Prediction
        prediction_scores_jpt = self.cls_jtp(sequence_output)

        # replace all labels higher than the JTP vocab size with the padding token '-100',
        # thus telling the loss function to ignore them
        labels_jtp = labels.where((labels < self.config_jtp.vocab_size) & (labels != -100), -100)

        jtp_loss = loss_fct(
            input=prediction_scores_jpt.view(-1, self.config_jtp.vocab_size), target=labels_jtp.view(-1)
        )
        jtp_loss = torch.nan_to_num(jtp_loss)

        # the total pre-training loss is the sum of the losses of both pre-training tasks
        loss = masked_lm_loss + jtp_loss

        output = MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.masked_lm_loss = masked_lm_loss
        output.jtp_loss = jtp_loss
        return output


class ARM64Tokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]',
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        self.preprocessor = Preprocessor(
            operand_formatters=(
                # 2-log numerical values and offsets to reduce the number of unique tokens we'll generate
                # (pre-made vocabulary used this too)
                operands.format_immediate_log,
                operands.format_offset_log,
            )
        )

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    def tokenize(self, texts, split_special_tokens=False, **kwargs):
        encoded_inputs = []
        for text in texts:
            cfg = dict(json.loads(text))
            tokens = self.preprocessor.preprocess(cfg)
            if len(tokens) < 512:
                tokens += [self.pad_token] * (512 - len(tokens))
            encoded_inputs.append(
                {
                    # The assembly preprocessor already splits the function into model vocabulary tokens.
                    'input_ids': self.convert_tokens_to_ids(tokens[:512]),
                }
            )

        return self.pad(
            encoded_inputs,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
