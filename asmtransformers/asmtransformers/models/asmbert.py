import json
from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss
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

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)

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

        loss_fct = CrossEntropyLoss()  # -100 index = padding token

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

        self.tokenizer = None
        self.padding = None

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

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = [self.tokenizer.pad_token] * 512

    def tokenize(self, texts, split_special_tokens=False, **kwargs):
        tokens_batch = []
        for text in texts:
            cfg = dict(json.loads(text))
            tokens = self.preprocessor.preprocess(cfg)
            if len(tokens) < 512:
                tokens += self.padding[: 512 - len(tokens)]
            tokens_batch.append(tokens)

        tokenized = self.tokenizer(
            tokens_batch,
            # input is pre-tokenized by the preprocessor
            is_split_into_words=True,
            add_special_tokens=False,
            # truncate the encoded sequence to a maximum of max_length tokens
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt',
        )
        return tokenized

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
