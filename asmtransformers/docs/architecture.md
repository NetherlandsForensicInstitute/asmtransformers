# Architecture Overview

## Purpose

`asmtransformers` is the training and inference core of the monorepo. It turns control-flow-graph representations of assembly functions into token sequences, feeds those sequences into transformer models, and exposes a script-oriented workflow for preprocessing, pretraining, finetuning, evaluation, and embedding generation.

This package is currently optimized for ARM64 assembly, but not every subsystem is ARM64-specific. The main architectural distinction is:

- ISA-specific logic lives in preprocessing and tokenization.
- Model composition, training flow, and most dataset handling are reusable patterns.

## Main Subsystems

### ISA preprocessing

The ISA-specific preprocessing path is implemented in [asmtransformers.arm64](../asmtransformers/arm64.py).

Its responsibilities are:

- parsing assembly instructions into opcode and operand tokens
- normalizing bracketed memory expressions and register sets into stable token streams
- identifying branch instructions and rewriting concrete jump targets into relative `JUMP_ADDR_*` tokens
- flattening a CFG-like input into a single model-ready token sequence

The central type is `Preprocessor`. It accepts either:

- a `dict[int, list[str]]` mapping basic-block offsets to instruction strings
- a `networkx.DiGraph` with an `asm` attribute on the (basic block) graph nodes.

The output is a flat token list suitable for a tokenizer or vocabulary builder.

### Operand normalization

Operand normalization helpers live in [asmtransformers.operands](../asmtransformers/operands.py).

These helpers reduce token explosion caused by raw numeric values. The current ARM64 path uses:

- `format_immediate_log()` for immediates such as `#0x1234`
- `format_offset_log()` for offsets such as `0x400`

The `Preprocessor` accepts `operand_formatters`, so normalization policy is a pluggable step rather than being hard-coded into parsing itself.

### Model wrappers

Model integration lives in [asmtransformers.models.asmbert](../asmtransformers/models/asmbert.py) and [asmtransformers.models.asmsentencebert](../asmtransformers/models/asmsentencebert.py).

There are two layers:

- `ASMBertForMaskedLM` and `ASMBertModel` adapt Hugging Face BERT classes to the jTrans-style setup, including shared word/position embeddings and jump-target prediction support during pretraining.
- `ASMSentenceTransformer` adapts the pretrained transformer into a sentence-transformers style embedding model for finetuning and inference.

The current tokenizer integration is ARM64-specific:

- `ARM64Tokenizer` owns the preprocessing step
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

- `preprocess.py`: tokenizes serialized CFG datasets
- `mkvocab.py`: builds a vocabulary from preprocessed assembly corpora
- `pretrain.py`: trains the masked-language-model / jump-target-prediction stage
- `finetune.py`: trains the embedding model for semantic similarity
- `evaluation.py`: evaluates retrieval quality
- `inference.py`: generates embeddings for downstream lookup workflows

These scripts are the package's practical orchestration layer. They define how datasets, tokenizers, and models are wired together during day-to-day research and development.

## End-to-End Data Flow

The current end-to-end flow is:

1. A function is represented as a serialized CFG where each block contains assembly instructions.
2. The ARM64 preprocessor parses instructions and operands and replaces direct branch targets with `JUMP_ADDR_*` tokens.
3. Operand formatters normalize large numeric values to reduce vocabulary growth.
4. A tokenizer converts the token stream into model inputs with the expected context length.
5. Pretraining uses those inputs for masked language modeling plus jump target prediction.
6. Finetuning wraps the transformer in a sentence-transformers pipeline and optimizes embedding similarity.
7. Inference encodes previously unseen functions into embeddings for downstream similarity search.

## What Is ARM64-Specific Today

The following parts are tightly coupled to ARM64:

- `asmtransformers.arm64.Preprocessor`
- ARM64 branch instruction lists and condition-code handling
- operand assumptions embodied in the current parser
- `ARM64Tokenizer`
- packaged model assets in `models/arm64bert/`
- scripts that directly instantiate `arm64.Preprocessor`

These elements would need to be replaced or generalized for additional instruction sets.

## What Is Reusable For Other ISAs Today

The following patterns are reusable across instruction sets:

- the general CFG-to-token-to-transformer pipeline
- the `Preprocessor` hook model for custom operand formatting
- Hugging Face BERT wrapping in `ASMBertModel` and `ASMBertForMaskedLM`
- sentence-transformers integration in `ASMSentenceTransformer`
- label-grouped dataset sampling in `LazySentenceLabelDataset`
- the script-level workflow stages: preprocess, vocab build, pretrain, finetune, evaluate, infer

In practice, reuse currently requires intentional adapter work because the public names and script wiring still assume ARM64.

## Testing And Regression Coverage

The current architecture is anchored by tests in:

- [tests/test_arm64.py](../tests/test_arm64.py) for parsing, tokenization, jump handling, and prefix-token behavior
- [tests/test_operand_formatters.py](../tests/test_operand_formatters.py) for numeric normalization behavior
- [tests/test_asmsentencebert.py](../tests/test_asmsentencebert.py) for model integration and embedding stability checks

Contributor changes that affect preprocessing, tokenization, or model composition should preserve the invariants covered there or extend the suite accordingly.
