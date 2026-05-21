# Pretraining

This page describes the current pretraining workflow for a from-scratch, multi-architecture ASMTransformers model on a
CUDA GPU cluster. The training entrypoint is `scripts/pretrain.py`; the SLURM template is
`scripts/slurm_pretrain.sh`.

## Inputs

Pretraining expects a Hugging Face dataset saved with `save_to_disk()` and loaded with `datasets.load_from_disk()`.
The training dataset should be a `DatasetDict` with:

- `train`: tokenized training functions.
- `test`: tokenized evaluation functions.

The pretraining script consumes tokenized rows with `input_ids` and `attention_mask`. For a raw multi-architecture
dataset, preprocess it first with `scripts/preprocess.py`. Raw rows must include:

- `cfg`: serialized control-flow graph JSON.
- `architecture`: one of the tokenizer-supported architectures: `arm64`, `amd64`, `i386`, or `riscv64`.

When `architecture` is absent, preprocessing falls back to `arm64` for compatibility with older ARM64-only datasets.

## From-Scratch Multi-Arch Run

Build or provide a tokenizer whose vocabulary covers all target architectures, then preprocess the corpus with that
tokenizer:

```bash
cd asmtransformers
pdm run python scripts/preprocess.py /path/to/tokenizer /path/to/raw-dataset /path/to/tokenized-dataset
```

Launch a single-node dry run before using the cluster. Keep the sample small and verify that the model starts training,
writes TensorBoard logs, and saves checkpoints:

```bash
cd asmtransformers
pdm run torchrun --nproc-per-node 1 scripts/pretrain.py \
    output \
    --data /path/to/tokenized-dataset \
    --tokenizer /path/to/tokenizer \
    --config asmtransformers/models/arm64bert/arm64bert_config.json \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --max-steps 20 \
    --save-steps 100 \
    --logging-steps 10 \
    --eval-samples 1000 \
    --bf16 \
    --tf32
```

For SLURM, edit the `#SBATCH` header in `scripts/slurm_pretrain.sh` for the target partition, account, wall time,
node count, and GPU count. Submit with dataset, tokenizer, and output paths:

```bash
cd asmtransformers
DATA=/path/to/tokenized-dataset \
TOKENIZER=/path/to/tokenizer \
OUTPUT_DIR=/path/to/output \
BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=4 \
sbatch scripts/slurm_pretrain.sh
```

The effective global batch size is:

```text
nodes * gpus_per_node * batch_size * gradient_accumulation_steps
```

For example, 2 nodes with 8 GPUs each, `BATCH_SIZE=16`, and `GRADIENT_ACCUMULATION_STEPS=4` gives a global batch size
of 1024 sequences.

## Precision

Use `--bf16` for CUDA bfloat16 mixed precision. This enables Hugging Face `TrainingArguments(bf16=True)`, so autocast
uses bfloat16 where appropriate while preserving normal optimizer and checkpoint behavior.

Use `--tf32` on Ampere, Hopper, or newer NVIDIA GPUs to allow TF32 matmul/cudnn kernels. The script enables PyTorch TF32
backend flags and passes TF32 through to `TrainingArguments`.

These flags are strict requests. Do not use them on CPU-only systems or older CUDA GPUs that do not support them; the
script exits before training rather than silently falling back. For local CPU-only dry runs or CI jobs, omit `--bf16` and
`--tf32`.

## Checkpoints And Resume

The script writes timestamped runs under the positional `output_dir`:

```text
output/pretraining_mlm_YYYY-MM-DD_HH-MM-SS/
```

It saves the tokenizer, periodic Trainer checkpoints, TensorBoard logs, and the final model. Keep checkpoint retention
bounded with `--save-total-limit`.

Resume an interrupted run from a Trainer checkpoint:

```bash
pdm run torchrun --nproc-per-node 8 scripts/pretrain.py \
    /path/to/output \
    --data /path/to/tokenized-dataset \
    --tokenizer /path/to/tokenizer \
    --resume-from-checkpoint /path/to/output/pretraining_mlm_.../checkpoint-10000 \
    --bf16 \
    --tf32
```

For a fresh multi-arch model, leave `--model-path` unset. Use `--model-path` only when continuing from a compatible
model checkpoint with the same tokenizer/vocabulary assumptions.
