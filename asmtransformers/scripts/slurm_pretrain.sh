#!/usr/bin/env bash
#SBATCH --job-name=asm-pretrain
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

DATA=${DATA:?set DATA to a preprocessed Hugging Face dataset directory}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the multi-arch tokenizer directory}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to the base output directory}
CONFIG=${CONFIG:-asmtransformers/models/multilingual_asmbert/config.json}

BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-10000}
LOGGING_STEPS=${LOGGING_STEPS:-100}
EVAL_SAMPLES=${EVAL_SAMPLES:-100000}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
MLM_PROB=${MLM_PROB:-0.15}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
SEED=${SEED:-42}
ASMTRANSFORMERS_RUN_ID=${ASMTRANSFORMERS_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_slurm_${SLURM_JOB_ID}}

GPUS_PER_NODE=${GPUS_PER_NODE}
NNODES=${NNODES}
NODE_RANK=${NODE_RANK}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export ASMTRANSFORMERS_RUN_ID

pdm run torchrun \
    --nnodes "$NNODES" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    scripts/pretrain.py \
    "$OUTPUT_DIR" \
    --data "$DATA" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG" \
    --epoch "$EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --logging-steps "$LOGGING_STEPS" \
    --eval-samples "$EVAL_SAMPLES" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --mlm-prob "$MLM_PROB" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --seed "$SEED" \
    --bf16 \
    --tf32
