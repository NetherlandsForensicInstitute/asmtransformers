# Architecture Overview

## Purpose

`asmtransformers` is the training and inference core of the monorepo. It turns control-flow-graph representations of assembly functions into token sequences, feeds those sequences into transformer models, and exposes a script-oriented workflow for preprocessing, pretraining, finetuning, evaluation, and embedding generation.

The current released model assets are optimized for ARM64 assembly, but not every subsystem is ARM64-specific. The main architectural distinction is:

- ISA-specific logic lives in preprocessing and tokenization.
- Model composition, training flow, and most dataset handling are reusable patterns.

## Main Subsystems

### ISA preprocessing

The ISA-specific preprocessing path is implemented in [asmtransformers.preprocessors](../asmtransformers/preprocessors/__init__.py), with current implementations for ARM64, x86/amd64, and RISC-V.

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

Model integration lives in [asmtransformers.models.asmbert](../asmtransformers/models/asmbert.py) and [asmtransformers.models.finetuning](../asmtransformers/models/finetuning.py).

The main layers are:

- `ASMBertForMaskedLM` and `ASMBertModel` adapt Hugging Face BERT classes to the jTrans-style setup, including shared word/position embeddings and jump-target prediction support during pretraining.
- `build_finetuning_model()` adapts the pretrained transformer into a native embedding model for triplet-loss finetuning.
- `ASMEmbedder` provides native inference for finetuned embedding checkpoints.

`build_finetuning_model()` returns a native `torch.nn.Module`. It does not provide the old SentenceTransformer methods
such as `.fit()` or `.save()`, and finetuned checkpoints are saved as `ASMBertModel` checkpoints rather than
SentenceTransformer module directories.

Tokenizer integration is handled by `ASMTokenizer`:

- it owns the architecture dispatch table for `amd64`, `arm64`, `i386`, and `riscv64`
- it converts serialized CFG input into padded token batches
- it uses the inherited `BertTokenizer` vocabulary and padding machinery for ID conversion

### Dataset and training helpers

Dataset helpers live in [asmtransformers.datasets.sentencelabel](../asmtransformers/datasets/sentencelabel.py).

`LazySentenceLabelDataset` bridges Hugging Face datasets and native triplet-loss training by:

- grouping rows by label
- lazily sampling multiple examples per label
- emitting plain `cfg` and `label` records suitable for triplet-style training

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
6. Finetuning wraps the transformer in a native embedding module and optimizes embedding similarity.
7. Inference uses the native embedder to encode previously unseen functions for downstream similarity search.

## What Is Still ARM64-Specific Today

The following parts are still ARM64-specific or ARM64-defaulted:

- packaged model assets in `models/arm64bert/`
- scripts and runtime paths that default to `arm64` when no architecture is supplied

Preprocessing itself is no longer ARM64-only: `ASMTokenizer` also dispatches to x86/amd64 and RISC-V preprocessors.
There is currently no packaged vocabulary or trained model for non-ARM64 ISAs.

## What Is Reusable For Other ISAs Today

The following patterns are reusable across instruction sets:

- the general CFG-to-token-to-transformer pipeline
- the `ASMPreprocessor` hook model for custom operand formatting
- `ASMTokenizer` architecture dispatch
- Hugging Face BERT wrapping in `ASMBertModel` and `ASMBertForMaskedLM`
- native triplet-loss finetuning integration via `build_finetuning_model()`
- native embedding inference in `ASMEmbedder`
- label-grouped dataset sampling in `LazySentenceLabelDataset`
- the script-level workflow stages: preprocess, vocab build, pretrain, finetune, evaluate, infer

In practice, reuse currently requires intentional adapter work because model assets and default runtime paths still assume ARM64.

## Testing And Regression Coverage

The current architecture is anchored by tests in:

- [tests/test_arm64.py](../tests/test_arm64.py) for parsing, tokenization, jump handling, and prefix-token behavior
- [tests/test_x86.py](../tests/test_x86.py) and [tests/test_riscv.py](../tests/test_riscv.py) for additional ISA preprocessing behavior
- [tests/test_operand_formatters.py](../tests/test_operand_formatters.py) for numeric normalization behavior
- [tests/test_asmbert.py](../tests/test_asmbert.py) for model integration and embedding stability checks
- [tests/test_finetuning.py](../tests/test_finetuning.py) for finetuning freeze policy and triplet-loss behavior
- [tests/test_embedder.py](../tests/test_embedder.py) for native embedding inference

Contributor changes that affect preprocessing, tokenization, or model composition should preserve the invariants covered there or extend the suite accordingly.
