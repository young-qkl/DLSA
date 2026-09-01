#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/RefSegRS}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/base_refsegrs}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py \
    --dataset refsegrs \
    --refsegrs_data_root "${DATA_ROOT}" \
    --model lavt_one \
    --model_id RMSIN_RefSegRS \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 60 \
    --lr 3e-5 \
    --batch-size 2 \
    --workers 4 \
    --img_size 480 \
    --swin_type base \
    --window12 \
    --pretrained_swin_weights "${ROOT}/pretrained_weights/swin_base_patch4_window12_384_22k.pth" \
    --ck_bert "${ROOT}/bert-base-uncased" \
    --bert_tokenizer "${ROOT}/bert-base-uncased" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
