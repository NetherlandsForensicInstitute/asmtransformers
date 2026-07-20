#!/usr/bin/env bash
#SBATCH --job-name=asm-pretrain
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

apptainer exec --nv --env TOKENIZER=/data/tokenizer/ \
                    --env DATA=/data/dataset/ \
                    --env OUTPUT_DIR=/data/output/ \
                    --env CONFIG=/app/asmtransformers/models/multilingual_asmbert/config.json \
                    --env GPUS_PER_NODE=$GPUS_PER_NODE \
                    --env NNODES=$NNODES \
                    --env NODE_RANK=$NODE_RANK \
                    --env MASTER_ADDR=$MASTER_ADDR \
                    --env MASTER_PORT=$MASTER_PORT \
                    --bind ./multiarch-dataset-20260611-tokenizer:/data/tokenizer \
                    --bind ./multiarch-dataset-20260611-tokenized-split/:/data/dataset \
                    --bind ./output/:/data/output  \
                    scs-pretrain.sif \
                    /app/scripts/slurm_pretrain.sh
