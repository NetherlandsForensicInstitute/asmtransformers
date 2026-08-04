#!/bin/bash
TORCHRUN_ARGS=(
    --nnodes "$NNODES"
    --nproc-per-node "$GPUS_PER_NODE" 
    --rdzv-backend "c10d" 
    --rdzv-endpoint "$MASTER_ADDR":"$MASTER_PORT" 
)

TRAINIG_ARGS=(
    --data "/data/dataset"
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
    "/data/output/" 
)

pdm run torchrun \
    "${TORCHRUN_ARGS[@]}" \
    scripts/pretrain.py \
    "${TRAINIG_ARGS[@]}" \

