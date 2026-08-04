# Architecture Overview

## Purpose

`asmtransformers` is the training and inference core of the monorepo. It turns control-flow-graph representations of 
assembly functions into token sequences, feeds those sequences into transformer models, and exposes a script-oriented 
workflow for preprocessing, pretraining, finetuning, evaluation, and embedding generation.

## Main Subsystems

### ISA preprocessing

The ISA-specific preprocessing path is implemented in [asmtransformers.preprocessors](../asmtransformers/preprocessors/__init__.py), with current implementations 
for amd64, arm64, i386 and riscv64 (though amd64 and i386 are preprocessed by the same `X86Preprocessor`).

Its responsibilities are:

- parsing assembly instructions into opcode and operand tokens
- normalizing bracketed memory expressions and register sets into stable token streams
- identifying branch instructions and rewriting concrete jump targets into relative `JUMP_ADDR_*` tokens
- flattening a CFG-like input into a single model-ready token sequence

The central type is `ASMPreprocessor`. It accepts:

- a `dict[int, list[str]]` mapping basic-block offsets to instruction strings

The output is a flat token list suitable for a tokenizer or vocabulary builder.

### Operand normalization

Operand normalization helpers live in [asmtransformers.operands](../asmtransformers/operands.py).

These helpers reduce token explosion caused by raw numeric values. The current tokenizer setup uses:

- `format_immediate_log()` for immediates such as `#0x1234`
- `format_offset_log()` for offsets such as `0x400`

`ASMPreprocessor` accepts `operand_formatters`, so normalization policy is a pluggable step rather than being hard-coded into parsing itself.

### Model wrappers

Model integration lives in [asmtransformers.models.asmbert](../asmtransformers/models/asmbert.py) and [asmtransformers.models.asmsentencebert](../asmtransformers/models/asmsentencebert.py).

The main layers are:

TO DO: I believe the below is up to date but let's check

- `ASMBertForMaskedLM` and `ASMBertModel` adapt Hugging Face BERT classes to the jTrans-style setup, including shared word/position embeddings and jump-target prediction support during pretraining.
- `ASMTransformerModule` adapts the pretrained transformer into a plain `SentenceTransformer` model for triplet-loss finetuning.
- `ASMEmbedder` provides native inference without requiring sentence-transformers at deployment time.

Tokenizer integration is handled by `ASMTokenizer`:

- it owns the architecture dispatch table for `amd64`, `arm64`, `i386`, and `riscv64`
- it converts serialized CFG input into padded token batches
- it uses the inherited `BertTokenizer` vocabulary and padding machinery for ID conversion

### Dataset and training helpers

Dataset helpers live in [asmtransformers.datasets.sentencelabel](../asmtransformers/datasets/sentencelabel.py).

`LazySentenceLabelDataset` bridges Hugging Face datasets and sentence-transformers training by:

- grouping rows by label
- lazily sampling multiple examples per label
- emitting `InputExample` objects suitable for triplet-style training

This layer is largely architecture-agnostic as long as the dataset schema remains consistent.

### Script-driven workflows

Operational entrypoints live in `asmtransformers/scripts/`. The most important ones are:

- `tokenize_dataset.py`: tokenizes serialized CFG datasets
- `mktokenizer.py`: builds a tokenizer vocabulary from assembly corpora
- `pretrain.py`: trains the masked-language-model / jump-target-prediction stage
- `finetune.py`: trains the embedding model for semantic similarity
- `evaluation.py`: evaluates retrieval quality
- `inference.py`: generates embeddings for downstream lookup workflows

These scripts are the package's practical orchestration layer. They define how datasets, tokenizers, and models are wired together during day-to-day research and development.

## End-to-End Data Flow

The current end-to-end flow is:

1. A function is represented as a serialized CFG where each block contains assembly instructions.
2. The selected ISA preprocessor parses instructions and operands and replaces direct branch targets with `JUMP_ADDR_*` tokens.
3. Operand formatters normalize large numeric values to reduce vocabulary growth.
4. A tokenizer converts the token stream into model inputs with the expected context length.
5. Pretraining uses those inputs for masked language modeling plus jump target prediction.
6. Finetuning wraps the transformer in a sentence-transformers pipeline (TODO is this still accurate?) and optimizes embedding similarity.
7. Inference uses the native embedder to encode previously unseen functions for downstream similarity search.

## What Is Still ARM64-Specific Today

The following parts are still ARM64-specific or ARM64-defaulted:

- packaged model assets in `models/arm64bert/`
- scripts and runtime paths that default to `arm64` when no architecture is supplied

TO DO: is the second line still correct?

Preprocessing itself is no longer ARM64-only: `ASMTokenizer` also dispatches to x86/amd64 and RISC-V preprocessors.
The multilingual model can be found on [Huggingface](https://huggingface.co/NetherlandsForensicInstitute/Multilingual-ASMBERT)

## What Is Reusable For Other ISAs Today

TO DO: is this correct? did I miss anything?

If you want to create your own model for a different assembly architecture using this repo, we recommend you take the
following steps:

- create an architecture-specific preprocessor in `models/preprocessors`
- ensure the architecture name is specified in `models/__init__.py`
- ensure it is specified in `ASMTokenizer` in `models/multilingual_asmbert/asmbert.py`

The multilingual code should work equally well when only data from one architecture is inserted.

## Testing And Regression Coverage

TO DO: this needs updating

The current architecture is anchored by tests in:

- [tests/test_arm64.py](../tests/test_arm64.py) for ARM64-specific parsing, tokenization, jump handling, and prefix-token behavior
- [tests/test_asmbert.py](../tests/test_asmbert.py) for model integration and embedding stability checks
- [tests/test_asmsentencebert_freeze.py](../tests/test_asmsentencebert_freeze.py) for finetuning freeze policy
- [tests/test_embedder.py](../tests/test_embedder.py) for native embedding inference
- [tests/test_evaluation.py](../tests/test_evaluation.py) for evaluation metrics and evaluation anchor/positive/negatives generation
- [tests/test_mktokenizer.py](../tests/test_mktokenizer.py) for extracting tokens
- [tests/test_operand_formatters.py](../tests/test_operand_formatters.py) for numeric normalization behavior
- [tests/test_pretrain.py](../tests/test_pretrain.py) for machine learning processes, training environment requirements (TODO: is that an accurate description?)
- [tests/test_riscv.py](../tests/test_riscv.py) for RISC-V specific parsing
- [tests/test_sentencelabel.py](../tests/test_sentencelabel.py) for Transformers-Sentence Transformers compatibility
- [tests/test_x86.py](../tests/test_x86.py) and x68-specific parsing
- 
- 


Contributor changes that affect preprocessing, tokenization, or model composition should preserve the invariants covered there or extend the suite accordingly.
