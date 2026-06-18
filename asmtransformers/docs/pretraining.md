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
dataset, tokenize it first with `scripts/tokenize_dataset.py`. Raw rows must include:

- `cfg`: serialized control-flow graph JSON.
- `architecture`: one of the tokenizer-supported architectures: `arm64`, `amd64`, `i386`, or `riscv64`.

When `architecture` is absent, preprocessing falls back to `arm64` for compatibility with older ARM64-only datasets.

## From-Scratch Multi-Arch Run

Build or provide a tokenizer whose vocabulary covers all target architectures, then preprocess the corpus with that
tokenizer:

```bash
cd asmtransformers
pdm run python scripts/tokenize_dataset.py /path/to/tokenizer /path/to/raw-dataset /path/to/tokenized-dataset
```

Launch a single-node dry run before using the cluster. Keep the sample small and verify that the model starts training,
writes TensorBoard logs, and saves checkpoints:

```bash
cd asmtransformers
pdm run torchrun --nproc-per-node 1 scripts/pretrain.py \
    output \
    --data /path/to/tokenized-dataset \
    --tokenizer /path/to/tokenizer \
    --config asmtransformers/models/multilingual_asmbert/config.json \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --max-steps 20 \
    --save-steps 100 \
    --logging-steps 10 \
    --eval-samples 1000 \
    --bf16 \
    --tf32
```

## Docker Container

The repository includes `pretrain.Containerfile` for building a pretraining image with Python, PDM, the
`asmtransformers` package, and the training scripts installed under `/app`.

Build the image from the `asmtransformers` folder that contains `pretrain.Containerfile`:

```bash
cd asmtransformers
docker build -f pretrain.Containerfile -t asmtransformers-pretrain .
```

Run a small CPU smoke test by bind-mounting the tokenized dataset, tokenizer, and output directory into the container.
Omit `--bf16` and `--tf32` for CPU-only runs:

```bash
mkdir -p output
docker run --rm \
    -v /path/to/tokenized-dataset:/data/tokenized-dataset:ro \
    -v /path/to/tokenizer:/data/tokenizer:ro \
    -v /path/to/output:/output \
    asmtransformers-pretrain \
    pdm run torchrun --nproc-per-node 1 /app/scripts/pretrain.py \
        /output \
        --data /data/tokenized-dataset \
        --tokenizer /data/tokenizer \
        --config /app/asmtransformers/models/multilingual_asmbert/config.json \
        --batch-size 1 \
        --gradient-accumulation-steps 1 \
        --max-steps 20 \
        --save-steps 100 \
        --logging-steps 10 \
        --eval-samples 1000
```

For a CUDA GPU run, install and configure the NVIDIA Container Toolkit on the Docker host, then expose GPUs with
`--gpus all`. Use `--bf16` and `--tf32` only when the exposed GPUs support those modes:

```bash
docker run --rm --gpus all \
    -v /path/to/tokenized-dataset:/data/tokenized-dataset:ro \
    -v /path/to/tokenizer:/data/tokenizer:ro \
    -v /path/to/output:/output \
    asmtransformers-pretrain \
    pdm run torchrun --nproc-per-node 1 /app/scripts/pretrain.py \
        /output \
        --data /data/tokenized-dataset \
        --tokenizer /data/tokenizer \
        --config /app/asmtransformers/models/multilingual_asmbert/config.json \
        --batch-size 16 \
        --gradient-accumulation-steps 4 \
        --save-steps 10000 \
        --logging-steps 100 \
        --eval-samples 100000 \
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

## Quality Annealing (Final Phase)

The last phase of pretraining oversamples high-quality functions so the model finishes on a corpus skewed toward
clean control-flow graphs. This is a separate, short run that continues from the phase-1 model.

First score the tokenized dataset (adds a `score` column per function), then build an oversampled training set:

```bash
cd asmtransformers
pdm run python scripts/quality_score.py /path/to/tokenized-dataset /path/to/tokenizer.json /path/to/scored-dataset
pdm run python scripts/oversample_quality.py /path/to/scored-dataset /path/to/annealing-dataset \
    --top-fraction 0.25 \
    --repeats 3
```

`oversample_quality.py` keeps every function once and adds `--repeats` extra copies of those in the top
`--top-fraction` of scores (so `--top-fraction 0.25 --repeats 3` quadruples the highest-quality quarter of the
corpus). The full corpus is retained, which limits forgetting; the `test` split is passed through unchanged. The
output is a pretrain-ready `DatasetDict`.

Run the annealing phase from the phase-1 model with a smaller learning rate and a short step budget. Launch `torchrun` against the local GPUs:

```bash
cd asmtransformers
pdm run torchrun --nproc-per-node 8 scripts/pretrain.py \
    /path/to/output \
    --model-path /path/to/output/pretraining_mlm_<phase-1>/ \
    --data /path/to/annealing-dataset \
    --tokenizer /path/to/tokenizer \
    --learning-rate 2e-5 \
    --warmup-ratio 0.0 \
    --epoch 1 \
    --run-id anneal \
    --bf16 \
    --tf32
```

Use `--model-path` (not `--resume-from-checkpoint`): this starts a fresh, short cosine schedule from the phase-1
weights rather than resuming phase-1's optimizer state and step count. Keep the learning rate below the
phase-1 LR.

## Precision

Use `--bf16` for CUDA bfloat16 mixed precision. This enables Hugging Face `TrainingArguments(bf16=True)`, so autocast
uses bfloat16 where appropriate while preserving normal optimizer and checkpoint behavior.

Use `--tf32` on Ampere, Hopper, or newer NVIDIA GPUs to allow TF32 matmul/cudnn kernels. The script enables PyTorch TF32
backend flags and passes TF32 through to `TrainingArguments`.

These flags are strict requests. Do not use them on CPU-only systems or older CUDA GPUs that do not support them; the
script exits before training rather than silently falling back. For local CPU-only dry runs or CI jobs, omit `--bf16` and
`--tf32`.

## Checkpoints And Resume

The script writes runs under the positional `output_dir`. Local runs use a timestamped directory:

```text
output/pretraining_mlm_YYYY-MM-DD_HH-MM-SS/
```

SLURM runs launched through `scripts/slurm_pretrain.sh` export one shared run id before `srun`, so all torchrun ranks
agree on the same output directory:

```text
output/pretraining_mlm_YYYY-MM-DD_HH-MM-SS_slurm_$SLURM_JOB_ID/
```

For repeated launches in the same scheduler allocation, non-SLURM multi-node launchers, or a custom run name, set
`ASMTRANSFORMERS_RUN_ID` or pass `--run-id`. The explicit `--run-id` takes precedence over `ASMTRANSFORMERS_RUN_ID`,
which takes precedence over `SLURM_JOB_ID`; otherwise the script falls back to a local timestamp.

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
