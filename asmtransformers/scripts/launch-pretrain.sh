#!/bin/bash
# This launch-pretrain script is here if you aim to pretrain on a SLURM controlled environment. When pretraining locally this script is not neccesairy. 
TORCHRUN_ARGS=(
    --nnodes "$NNODES"
    --nproc-per-node "$GPUS_PER_NODE" 
    --rdzv-backend "c10d" 
    --rdzv-endpoint "$MASTER_ADDR":"$MASTER_PORT" 
)
# All arguments below are hardcoded here, they should be an ENV variable so that they can be set from outside the container.
# This is recommended for anyone who aims to run pretrain.py from a container.
TRAINIG_ARGS=(
    --data "/data/dataset" # When executing this script from a container, map the dataset and tokenizer volume to /data/dataset and /data/tokenizer
    --tokenizer "/data/tokenizer"
    --config "/app/asmtransformers/models/multilingual_asmbert/config_large.json"
    --epoch 19
    --batch-size 192
    --gradient-accumulation-steps 4
    --mlm-prob 0.4
    --save-steps 5000
    --dataloader-num-workers 4
    --bf16
    --tf32
    "/data/output/" # This also needs to be mapped to a volume when spinning up the container
)

pdm run torchrun \
    "${TORCHRUN_ARGS[@]}" \
    scripts/pretrain.py \
    "${TRAINIG_ARGS[@]}" \

